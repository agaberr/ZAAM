from flask import jsonify, render_template


# Route to check if the API is running (for render deployment)
def register_main_routes(app):
    @app.route('/api/status')
    def status():
        return jsonify({"status": "running", "message": "ZAAM API is running"}) 
