from flask import jsonify, render_template

def register_main_routes(app):
    @app.route('/api/status')
    def api_status():
        return jsonify({"status": "running", "message": "ZAAM API is running"}) 
