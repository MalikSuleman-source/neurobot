from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from groq import Groq
import sqlite3
import os
import json
import time
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fallback-key')

# ─── GROQ CLIENT ────────────────────────────────────────────────
client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

# ─── LOGIN MANAGER ──────────────────────────────────────────────
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'

# ─── AVAILABLE MODELS ───────────────────────────────────────────
MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 · 70B",
    "llama-3.1-8b-instant":    "Llama 3.1 · 8B Fast",
    "mixtral-8x7b-32768":      "Mixtral · 8x7B",
    "gemma2-9b-it":            "Gemma 2 · 9B",
}

SYSTEM_PROMPT = """You are NeuroBot, an advanced AI assistant.
You are helpful, intelligent, and friendly.
You remember the conversation history and give contextual responses.
Keep responses clear and concise unless asked for detail."""

# ─── RATE LIMITING ──────────────────────────────────────────────
request_counts = {}
RATE_LIMIT = 20       # max requests
RATE_WINDOW = 60      # per 60 seconds

def rate_limit_check(user_id):
    now = time.time()
    key = str(user_id)
    if key not in request_counts:
        request_counts[key] = []
    # Remove old entries
    request_counts[key] = [t for t in request_counts[key] if now - t < RATE_WINDOW]
    if len(request_counts[key]) >= RATE_LIMIT:
        return False
    request_counts[key].append(now)
    return True

# ─── DATABASE SETUP ─────────────────────────────────────────────
def get_db():
    db = sqlite3.connect('neurobot.db')
    db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    db.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            title TEXT,
            model TEXT DEFAULT "llama-3.3-70b-versatile",
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            tokens_approx INTEGER,
            response_time_ms INTEGER,
            language TEXT DEFAULT "en",
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS reactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            reaction TEXT NOT NULL,
            FOREIGN KEY (message_id) REFERENCES messages(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    ''')
    db.commit()
    db.close()

init_db()

# ─── USER MODEL ─────────────────────────────────────────────────
class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    db.close()
    if user:
        return User(user['id'], user['username'], user['email'])
    return None

# ─── AUTH ROUTES ────────────────────────────────────────────────
@app.route('/login_page')
def login_page():
    return render_template('auth.html', mode='login')

@app.route('/register_page')
def register_page():
    return render_template('auth.html', mode='register')

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username', '').strip()
    email    = data.get('email', '').strip()
    password = data.get('password', '')

    if not username or not email or not password:
        return jsonify({'error': 'All fields required'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    db = get_db()
    try:
        db.execute(
            'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
            (username, email, generate_password_hash(password))
        )
        db.commit()
        user_row = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        user = User(user_row['id'], user_row['username'], user_row['email'])
        login_user(user)
        return jsonify({'status': 'ok', 'username': username})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username or email already exists'}), 409
    finally:
        db.close()

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')

    db = get_db()
    user_row = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    db.close()

    if user_row and check_password_hash(user_row['password_hash'], password):
        user = User(user_row['id'], user_row['username'], user_row['email'])
        login_user(user)
        return jsonify({'status': 'ok', 'username': username})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'status': 'logged_out'})

@app.route('/api/me')
@login_required
def me():
    return jsonify({'id': current_user.id, 'username': current_user.username, 'email': current_user.email})

# ─── MAIN ROUTES ────────────────────────────────────────────────
@app.route('/')
@login_required
def home():
    return render_template('index.html', username=current_user.username)

# ─── CHAT SESSIONS ──────────────────────────────────────────────
@app.route('/api/sessions')
@login_required
def get_sessions():
    db = get_db()
    rows = db.execute(
        'SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT 30',
        (current_user.id,)
    ).fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/sessions/<session_id>/messages')
@login_required
def get_session_messages(session_id):
    db = get_db()
    rows = db.execute(
        'SELECT * FROM messages WHERE session_id = ? AND user_id = ? ORDER BY created_at ASC',
        (session_id, current_user.id)
    ).fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
@login_required
def delete_session(session_id):
    db = get_db()
    db.execute('DELETE FROM messages WHERE session_id = ? AND user_id = ?', (session_id, current_user.id))
    db.execute('DELETE FROM sessions WHERE session_id = ? AND user_id = ?', (session_id, current_user.id))
    db.commit()
    db.close()
    return jsonify({'status': 'deleted'})

# ─── CHAT ───────────────────────────────────────────────────────
@app.route('/chat', methods=['POST'])
@login_required
def chat():
    # Rate limit check
    if not rate_limit_check(current_user.id):
        return jsonify({'error': 'Rate limit exceeded. Please wait a minute.'}), 429

    data = request.json
    user_message = data.get('message', '').strip()
    session_id   = data.get('session_id', '')
    model        = data.get('model', 'llama-3.3-70b-versatile')
    language     = data.get('language', 'en')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    if model not in MODELS:
        model = 'llama-3.3-70b-versatile'

    # Language instruction
    lang_prompts = {
        'ur': 'Please respond in Urdu language.',
        'en': '',
        'auto': 'Respond in the same language the user used.',
    }
    lang_instruction = lang_prompts.get(language, '')

    db = get_db()

    # Create session if new
    existing = db.execute('SELECT id FROM sessions WHERE session_id = ? AND user_id = ?',
                          (session_id, current_user.id)).fetchone()
    if not existing:
        db.execute(
            'INSERT INTO sessions (user_id, session_id, title, model) VALUES (?, ?, ?, ?)',
            (current_user.id, session_id, user_message[:50], model)
        )
    else:
        db.execute(
            'UPDATE sessions SET updated_at = CURRENT_TIMESTAMP, model = ? WHERE session_id = ? AND user_id = ?',
            (model, session_id, current_user.id)
        )

    # Save user message
    db.execute(
        'INSERT INTO messages (session_id, user_id, role, content) VALUES (?, ?, ?, ?)',
        (session_id, current_user.id, 'user', user_message)
    )
    db.commit()

    # Build message history
    history_rows = db.execute(
        'SELECT role, content FROM messages WHERE session_id = ? AND user_id = ? ORDER BY created_at ASC',
        (session_id, current_user.id)
    ).fetchall()
    db.close()

    system = SYSTEM_PROMPT
    if lang_instruction:
        system += f'\n{lang_instruction}'

    messages = [{'role': 'system', 'content': system}]
    messages += [{'role': r['role'], 'content': r['content']} for r in history_rows]

    def generate():
        full_response = ""
        start_ms = int(time.time() * 1000)

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
                stream=True
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    yield f"data: {json.dumps({'token': delta})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        elapsed_ms = int(time.time() * 1000) - start_ms

        # Save AI response + analytics
        db2 = get_db()
        msg_result = db2.execute(
            'INSERT INTO messages (session_id, user_id, role, content) VALUES (?, ?, ?, ?)',
            (session_id, current_user.id, 'assistant', full_response)
        )
        msg_id = msg_result.lastrowid

        db2.execute(
            'INSERT INTO analytics (user_id, model, tokens_approx, response_time_ms, language) VALUES (?, ?, ?, ?, ?)',
            (current_user.id, model, len(full_response.split()), elapsed_ms, language)
        )
        db2.commit()

        count = db2.execute('SELECT COUNT(*) as c FROM messages WHERE session_id = ? AND user_id = ?',
                            (session_id, current_user.id)).fetchone()['c']
        db2.close()

        yield f"data: {json.dumps({'done': True, 'message_count': count, 'msg_id': msg_id, 'elapsed_ms': elapsed_ms})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

# ─── CLEAR SESSION ──────────────────────────────────────────────
@app.route('/clear', methods=['POST'])
@login_required
def clear_chat():
    session_id = request.json.get('session_id', '')
    db = get_db()
    db.execute('DELETE FROM messages WHERE session_id = ? AND user_id = ?', (session_id, current_user.id))
    db.commit()
    db.close()
    return jsonify({'status': 'cleared'})

# ─── REACTIONS ──────────────────────────────────────────────────
@app.route('/api/react', methods=['POST'])
@login_required
def react():
    data = request.json
    msg_id   = data.get('message_id')
    reaction = data.get('reaction')  # 'thumbs_up' | 'thumbs_down'

    db = get_db()
    existing = db.execute(
        'SELECT id FROM reactions WHERE message_id = ? AND user_id = ?',
        (msg_id, current_user.id)
    ).fetchone()

    if existing:
        db.execute('UPDATE reactions SET reaction = ? WHERE message_id = ? AND user_id = ?',
                   (reaction, msg_id, current_user.id))
    else:
        db.execute('INSERT INTO reactions (message_id, user_id, reaction) VALUES (?, ?, ?)',
                   (msg_id, current_user.id, reaction))
    db.commit()
    db.close()
    return jsonify({'status': 'ok'})

# ─── ANALYTICS ──────────────────────────────────────────────────
@app.route('/api/analytics')
@login_required
def analytics():
    db = get_db()

    total_msgs = db.execute(
        'SELECT COUNT(*) as c FROM messages WHERE user_id = ?', (current_user.id,)
    ).fetchone()['c']

    total_sessions = db.execute(
        'SELECT COUNT(*) as c FROM sessions WHERE user_id = ?', (current_user.id,)
    ).fetchone()['c']

    model_usage = db.execute(
        'SELECT model, COUNT(*) as cnt FROM analytics WHERE user_id = ? GROUP BY model ORDER BY cnt DESC',
        (current_user.id,)
    ).fetchall()

    avg_response = db.execute(
        'SELECT AVG(response_time_ms) as avg FROM analytics WHERE user_id = ?', (current_user.id,)
    ).fetchone()['avg']

    daily = db.execute(
        '''SELECT DATE(created_at) as day, COUNT(*) as cnt
           FROM messages WHERE user_id = ? AND role = "user"
           GROUP BY day ORDER BY day DESC LIMIT 7''',
        (current_user.id,)
    ).fetchall()

    lang_usage = db.execute(
        'SELECT language, COUNT(*) as cnt FROM analytics WHERE user_id = ? GROUP BY language',
        (current_user.id,)
    ).fetchall()

    db.close()

    return jsonify({
        'total_messages': total_msgs,
        'total_sessions': total_sessions,
        'model_usage': [dict(r) for r in model_usage],
        'avg_response_ms': round(avg_response or 0),
        'daily_messages': [dict(r) for r in daily],
        'language_usage': [dict(r) for r in lang_usage],
    })

# ─── EXPORT CHAT ────────────────────────────────────────────────
@app.route('/api/export/<session_id>')
@login_required
def export_chat(session_id):
    fmt = request.args.get('format', 'txt')  # 'txt' or 'json'
    db = get_db()
    rows = db.execute(
        'SELECT role, content, created_at FROM messages WHERE session_id = ? AND user_id = ? ORDER BY created_at ASC',
        (session_id, current_user.id)
    ).fetchall()
    sess = db.execute('SELECT title FROM sessions WHERE session_id = ? AND user_id = ?',
                      (session_id, current_user.id)).fetchone()
    db.close()

    title = sess['title'] if sess else 'Chat Export'

    if fmt == 'json':
        data = {'title': title, 'session_id': session_id, 'messages': [dict(r) for r in rows]}
        return Response(
            json.dumps(data, indent=2),
            mimetype='application/json',
            headers={'Content-Disposition': f'attachment; filename="neurobot_{session_id[:8]}.json"'}
        )
    else:
        lines = [f"NeuroBot Chat Export — {title}", f"Session: {session_id}", "=" * 50, ""]
        for r in rows:
            role = "You" if r['role'] == 'user' else "NeuroBot"
            lines.append(f"[{r['created_at']}] {role}:")
            lines.append(r['content'])
            lines.append("")
        return Response(
            "\n".join(lines),
            mimetype='text/plain',
            headers={'Content-Disposition': f'attachment; filename="neurobot_{session_id[:8]}.txt"'}
        )

# ─── MODELS LIST ────────────────────────────────────────────────
@app.route('/api/models')
def get_models():
    return jsonify(MODELS)

if __name__ == '__main__':
    app.run(debug=True)
