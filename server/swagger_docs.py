from flask import Flask
from flask_restx import Api, Resource, fields, Namespace
from werkzeug.middleware.proxy_fix import ProxyFix

# Create Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Create API with documentation
api = Api(
    app, 
    version='1.0', 
    title='Reminder API',
    description='A reminder API with Google Calendar integration',
    doc='/api/docs',
    validate=True
)

# Create namespaces
auth_ns = Namespace('Authentication', description='Authentication operations', path='/api/auth')
user_ns = Namespace('Users', description='User operations', path='/api/users')
reminder_ns = Namespace('Reminders', description='Reminder operations', path='/api/reminders')
google_ns = Namespace('Google Integration', description='Google Calendar integration', path='/api/auth/google')

# Add namespaces to API
api.add_namespace(auth_ns)
api.add_namespace(user_ns)
api.add_namespace(reminder_ns)
api.add_namespace(google_ns)

# Define models for documentation
# Authentication models
register_model = auth_ns.model('Register', {
    'full_name': fields.String(required=True, description='User full name', example='John Doe'),
    'age': fields.Integer(description='User age', example=30),
    'gender': fields.String(description='User gender', example='Male'),
    'contact_info': fields.Raw(required=True, description='User contact information', example={
        'email': 'user@example.com',
        'phone': '+1234567890'
    }),
    'password': fields.String(required=True, description='User password', example='Password123'),
    'emergency_contacts': fields.List(fields.Raw, description='Emergency contacts', example=[
        {'name': 'Jane Doe', 'phone': '+1987654321', 'relationship': 'Spouse'}
    ])
})

login_model = auth_ns.model('Login', {
    'email': fields.String(required=True, description='User email', example='user@example.com'),
    'password': fields.String(required=True, description='User password', example='Password123')
})

token_model = auth_ns.model('Token', {
    'token': fields.String(description='JWT Token', example='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'),
    'user_id': fields.String(description='User ID', example='60d6ec9f2a57c123456789ab')
})

# User models
user_model = user_ns.model('User', {
    '_id': fields.String(description='User ID', example='60d6ec9f2a57c123456789ab'),
    'full_name': fields.String(description='User full name', example='John Doe'),
    'age': fields.Integer(description='User age', example=30),
    'gender': fields.String(description='User gender', example='Male'),
    'contact_info': fields.Raw(description='User contact information', example={
        'email': 'user@example.com',
        'phone': '+1234567890'
    }),
    'emergency_contacts': fields.List(fields.Raw, description='Emergency contacts', example=[
        {'name': 'Jane Doe', 'phone': '+1987654321', 'relationship': 'Spouse'}
    ]),
    'created_at': fields.DateTime(description='Creation timestamp', example='2023-06-10T09:30:00Z'),
    'updated_at': fields.DateTime(description='Last update timestamp', example='2023-06-10T09:30:00Z')
})

update_user_model = user_ns.model('UpdateUser', {
    'full_name': fields.String(description='User full name', example='John Doe'),
    'age': fields.Integer(description='User age', example=30),
    'gender': fields.String(description='User gender', example='Male'),
    'contact_info': fields.Raw(description='User contact information', example={
        'email': 'user@example.com',
        'phone': '+1234567890'
    }),
    'emergency_contacts': fields.List(fields.Raw, description='Emergency contacts', example=[
        {'name': 'Jane Doe', 'phone': '+1987654321', 'relationship': 'Spouse'}
    ])
})

# Reminder models
reminder_input_model = reminder_ns.model('ReminderInput', {
    'title': fields.String(required=True, description='Reminder title', example='Doctor Appointment'),
    'description': fields.String(description='Reminder description', example='Visit Dr. Smith for checkup'),
    'start_time': fields.DateTime(required=True, description='Start time in ISO format', 
                                example='2023-06-15T14:30:00Z'),
    'end_time': fields.DateTime(description='End time in ISO format', example='2023-06-15T15:30:00Z'),
    'recurrence': fields.String(description='Recurrence pattern', enum=['daily', 'weekly', 'monthly', 'yearly'], 
                              example='weekly')
})

reminder_output_model = reminder_ns.model('ReminderOutput', {
    '_id': fields.String(description='Reminder ID', example='60d6ec9f2a57c123456789ab'),
    'user_id': fields.String(description='User ID', example='60d6ec9f2a57c123456789ab'),
    'title': fields.String(description='Reminder title', example='Doctor Appointment'),
    'description': fields.String(description='Reminder description', example='Visit Dr. Smith for checkup'),
    'start_time': fields.DateTime(description='Start time', example='2023-06-15T14:30:00Z'),
    'end_time': fields.DateTime(description='End time', example='2023-06-15T15:30:00Z'),
    'recurrence': fields.String(description='Recurrence pattern', example='weekly'),
    'completed': fields.Boolean(description='Completion status', example=False),
    'google_event_id': fields.String(description='Google Calendar Event ID', 
                                  example='_6t13cdi6as32ba91hg32ba91j0'),
    'created_at': fields.DateTime(description='Creation timestamp', example='2023-06-10T09:30:00Z'),
    'updated_at': fields.DateTime(description='Last update timestamp', example='2023-06-10T09:30:00Z')
})

reminders_list_model = reminder_ns.model('RemindersList', {
    'reminders': fields.List(fields.Nested(reminder_output_model))
})

reminder_stats_model = reminder_ns.model('ReminderStats', {
    'completion_rate': fields.Float(description='Completion rate percentage', example=75.5),
    'today_count': fields.Integer(description='Number of reminders for today', example=3),
    'upcoming_count': fields.Integer(description='Number of upcoming reminders', example=12)
})

detailed_reminder_stats_model = reminder_ns.model('DetailedReminderStats', {
    'total_reminders': fields.Integer(description='Total number of reminders', example=50),
    'completed_reminders': fields.Integer(description='Number of completed reminders', example=35),
    'completion_rate': fields.Float(description='Completion rate percentage', example=70.0),
    'today_reminders': fields.Integer(description='Number of reminders for today', example=5),
    'today_completed': fields.Integer(description='Number of completed reminders for today', example=2),
    'today_completion_rate': fields.Float(description='Today\'s completion rate percentage', example=40.0),
    'upcoming_reminders': fields.Integer(description='Number of upcoming reminders (next 7 days)', example=15),
    'overdue_reminders': fields.Integer(description='Number of overdue reminders', example=3),
    'reminders_by_type': fields.Raw(description='Distribution of reminders by recurrence type', example={
        'one_time': 20,
        'daily': 10,
        'weekly': 15,
        'monthly': 5,
        'yearly': 0
    }),
    'recently_created': fields.Integer(description='Number of reminders created in the last 30 days', example=12)
})

reminder_id_model = reminder_ns.model('ReminderID', {
    'reminder_id': fields.String(description='ID of the created reminder', example='60d6ec9f2a57c123456789ab')
})

google_auth_url_model = google_ns.model('GoogleAuthURL', {
    'authorization_url': fields.String(description='Google OAuth authorization URL', 
                                    example='https://accounts.google.com/o/oauth2/auth?...')
})

google_status_model = google_ns.model('GoogleConnectionStatus', {
    'connected': fields.Boolean(description='Whether Google Calendar is connected', example=True),
    'connected_at': fields.DateTime(description='When the connection was established', 
                                 example='2023-06-10T09:30:00Z')
})

# Authentication routes
@auth_ns.route('/register')
class UserRegister(Resource):
    @auth_ns.doc('user_register')
    @auth_ns.expect(register_model)
    @auth_ns.response(201, 'User registered successfully')
    @auth_ns.response(400, 'Validation error')
    def post(self):
        """Register a new user"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in auth_routes.py'}

@auth_ns.route('/login')
class UserLogin(Resource):
    @auth_ns.doc('user_login')
    @auth_ns.expect(login_model)
    @auth_ns.response(200, 'Success', token_model)
    @auth_ns.response(401, 'Invalid credentials')
    def post(self):
        """Login and get authentication token"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in auth_routes.py'}

# User routes
@user_ns.route('')
class UserList(Resource):
    @user_ns.doc('create_user', security='apiKey')
    @user_ns.expect(register_model)
    @user_ns.response(201, 'User created successfully')
    @user_ns.response(400, 'Validation error')
    @user_ns.response(401, 'Authentication required')
    def post(self):
        """Create a new user (admin only)"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in user_routes.py'}

@user_ns.route('/<user_id>')
@user_ns.param('user_id', 'The user identifier')
class UserResource(Resource):
    @user_ns.doc('get_user', security='apiKey')
    @user_ns.response(200, 'Success', user_model)
    @user_ns.response(401, 'Authentication required')
    @user_ns.response(403, 'Unauthorized')
    @user_ns.response(404, 'User not found')
    def get(self, user_id):
        """Get a specific user by ID"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in user_routes.py'}
    
    @user_ns.doc('update_user', security='apiKey')
    @user_ns.expect(update_user_model)
    @user_ns.response(200, 'User updated successfully')
    @user_ns.response(400, 'Validation error')
    @user_ns.response(401, 'Authentication required')
    @user_ns.response(403, 'Unauthorized')
    @user_ns.response(404, 'User not found')
    def put(self, user_id):
        """Update a user"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in user_routes.py'}
    
    @user_ns.doc('delete_user', security='apiKey')
    @user_ns.response(200, 'User deleted successfully')
    @user_ns.response(401, 'Authentication required')
    @user_ns.response(403, 'Unauthorized')
    @user_ns.response(404, 'User not found')
    def delete(self, user_id):
        """Delete a user"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in user_routes.py'}

# Google OAuth routes
@google_ns.route('/connect')
class GoogleConnect(Resource):
    @google_ns.doc('google_connect', security='apiKey')
    @google_ns.response(200, 'Success', google_auth_url_model)
    @google_ns.response(401, 'Authentication required')
    def get(self):
        """Start Google OAuth flow by redirecting to Google's auth page"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in google_auth_routes.py'}

@google_ns.route('/status')
class GoogleStatus(Resource):
    @google_ns.doc('google_status', security='apiKey')
    @google_ns.response(200, 'Success', google_status_model)
    @google_ns.response(401, 'Authentication required')
    def get(self):
        """Check if user has connected Google account"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in google_auth_routes.py'}

@google_ns.route('/disconnect')
class GoogleDisconnect(Resource):
    @google_ns.doc('google_disconnect', security='apiKey')
    @google_ns.response(200, 'Success')
    @google_ns.response(401, 'Authentication required')
    @google_ns.response(500, 'Failed to disconnect')
    def post(self):
        """Disconnect Google account"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in google_auth_routes.py'}

@google_ns.route('/callback')
class GoogleCallback(Resource):
    @google_ns.doc('google_callback')
    @google_ns.param('state', 'OAuth state')
    @google_ns.param('code', 'OAuth authorization code')
    @google_ns.response(302, 'Redirect to success or failure page')
    def get(self):
        """Handle callback from Google OAuth (documentation only)"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in google_auth_routes.py'}

# Reminder routes
@reminder_ns.route('')
class ReminderList(Resource):
    @reminder_ns.doc('list_reminders', security='apiKey')
    @reminder_ns.response(200, 'Success', [reminder_output_model])
    @reminder_ns.response(401, 'Authentication required')
    @reminder_ns.param('completed', 'Filter by completion status (true/false)')
    def get(self):
        """Get all reminders for authenticated user"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in reminder_routes.py'}
    
    @reminder_ns.doc('create_reminder', security='apiKey')
    @reminder_ns.expect(reminder_input_model)
    @reminder_ns.response(201, 'Reminder created', reminder_id_model)
    @reminder_ns.response(400, 'Validation error')
    @reminder_ns.response(401, 'Authentication required')
    def post(self):
        """Create a new reminder"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in reminder_routes.py'}

@reminder_ns.route('/<reminder_id>')
@reminder_ns.param('reminder_id', 'The reminder identifier')
class ReminderResource(Resource):
    @reminder_ns.doc('get_reminder', security='apiKey')
    @reminder_ns.response(200, 'Success', reminder_output_model)
    @reminder_ns.response(401, 'Authentication required')
    @reminder_ns.response(403, 'Unauthorized')
    @reminder_ns.response(404, 'Reminder not found')
    def get(self, reminder_id):
        """Get a specific reminder by ID"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in reminder_routes.py'}
    
    @reminder_ns.doc('update_reminder', security='apiKey')
    @reminder_ns.expect(reminder_input_model)
    @reminder_ns.response(200, 'Reminder updated')
    @reminder_ns.response(400, 'Validation error')
    @reminder_ns.response(401, 'Authentication required')
    @reminder_ns.response(403, 'Unauthorized')
    @reminder_ns.response(404, 'Reminder not found')
    def put(self, reminder_id):
        """Update a reminder"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in reminder_routes.py'}
    
    @reminder_ns.doc('delete_reminder', security='apiKey')
    @reminder_ns.response(200, 'Reminder deleted')
    @reminder_ns.response(401, 'Authentication required')
    @reminder_ns.response(403, 'Unauthorized')
    @reminder_ns.response(404, 'Reminder not found')
    @reminder_ns.response(500, 'Deletion failed')
    def delete(self, reminder_id):
        """Delete a reminder"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in reminder_routes.py'}

@reminder_ns.route('/today')
class TodayReminders(Resource):
    @reminder_ns.doc('get_today_reminders', security='apiKey')
    @reminder_ns.response(200, 'Success', [reminder_output_model])
    @reminder_ns.response(401, 'Authentication required')
    def get(self):
        """Get reminders for today"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in reminder_routes.py'}

@reminder_ns.route('/upcoming')
class UpcomingReminders(Resource):
    @reminder_ns.doc('get_upcoming_reminders', security='apiKey')
    @reminder_ns.response(200, 'Success', [reminder_output_model])
    @reminder_ns.response(401, 'Authentication required')
    @reminder_ns.param('days', 'Number of days to look ahead (default: 7)')
    def get(self):
        """Get upcoming reminders within specified days"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in reminder_routes.py'}

@reminder_ns.route('/stats')
class ReminderStats(Resource):
    @reminder_ns.doc('get_reminder_stats', security='apiKey')
    @reminder_ns.response(200, 'Success', reminder_stats_model)
    @reminder_ns.response(401, 'Authentication required')
    def get(self):
        """Get reminder statistics"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in reminder_routes.py'}

@reminder_ns.route('/stats/detailed')
class DetailedReminderStats(Resource):
    @reminder_ns.doc('get_detailed_reminder_stats', security='apiKey')
    @reminder_ns.response(200, 'Success', detailed_reminder_stats_model)
    @reminder_ns.response(401, 'Authentication required')
    def get(self):
        """Get detailed reminder statistics including types, overdue counts, and more"""
        return {'message': 'This is a documentation endpoint. See the actual implementation in reminder_routes.py'}

# Security definitions
authorizations = {
    'apiKey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'Authorization',
        'description': 'Add a JWT token to the header with the format: Bearer {token}'
    }
}

api.authorizations = authorizations

if __name__ == '__main__':
    app.run(debug=True, port=5001) 