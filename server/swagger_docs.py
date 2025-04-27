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
    title='ZAAM API',
    description='ZAAM API for Memory Aid and Conversational AI',
    doc='/api/docs',
    validate=True
)

# Create namespaces
auth_ns = Namespace('Authentication', description='Authentication operations', path='/api/auth')
user_ns = Namespace('Users', description='User operations', path='/api/users')
memory_aid_ns = Namespace('Memory Aids', description='Memory aid operations', path='/api/memory-aids')
ai_ns = Namespace('AI Assistant', description='AI natural language processing', path='/api/ai')
qa_ns = api.namespace('api/qa', description='Conversational QA operations')

# Add namespaces to API
api.add_namespace(auth_ns)
api.add_namespace(user_ns)
api.add_namespace(memory_aid_ns)
api.add_namespace(ai_ns)
api.add_namespace(qa_ns)

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

# AI Assistant models
ai_input_model = ai_ns.model('AIInput', {
    'text': fields.String(required=True, description='Natural language input to process', 
                        example='What is the weather like? Any news about tech?')
})

ai_response_model = ai_ns.model('AIResponse', {
    'response': fields.String(description='AI generated response', 
                           example='WEATHER: Here\'s the weather information you asked about:\n- What is the weather like?\n\nNEWS: I found news information in your request:\n- Any news about tech?'),
    'success': fields.Boolean(description='Whether the request was successful', example=True)
})

category_response_model = ai_ns.model('CategoryResponse', {
    'response': fields.String(description='Category-specific response'),
    'category': fields.String(description='Category of the processed text', enum=['news', 'weather']),
    'success': fields.Boolean(description='Whether the request was successful', example=True)
})

# Memory Aid models
memory_aid_input_model = memory_aid_ns.model('MemoryAidInput', {
    'title': fields.String(required=True, description='Memory aid title', example='Sarah'),
    'description': fields.String(description='Memory aid description', 
                               example='My daughter who visits every Sunday. She has two children named Maya and Rohan.'),
    'type': fields.String(required=True, description='Memory aid type', 
                        enum=['person', 'place', 'event', 'object'], example='person'),
    'date': fields.String(description='Date associated with the memory aid', example='2023-05-15'),
    'image_url': fields.String(description='URL to an image for the memory aid', 
                             example='https://example.com/images/sarah.jpg')
})

memory_aid_output_model = memory_aid_ns.model('MemoryAidOutput', {
    '_id': fields.String(description='Memory aid ID', example='60d6ec9f2a57c123456789ab'),
    'user_id': fields.String(description='User ID', example='60d6ec9f2a57c123456789ab'),
    'title': fields.String(description='Memory aid title', example='Sarah'),
    'description': fields.String(description='Memory aid description', 
                               example='My daughter who visits every Sunday. She has two children named Maya and Rohan.'),
    'type': fields.String(description='Memory aid type', example='person'),
    'date': fields.String(description='Date associated with the memory aid', example='2023-05-15'),
    'image_url': fields.String(description='URL to an image for the memory aid', 
                             example='https://example.com/images/sarah.jpg'),
    'created_at': fields.DateTime(description='Creation timestamp', example='2023-06-10T09:30:00Z'),
    'updated_at': fields.DateTime(description='Last update timestamp', example='2023-06-10T09:30:00Z')
})

memory_aid_search_model = memory_aid_ns.model('MemoryAidSearch', {
    'query': fields.String(required=True, description='Search query', example='daughter')
})

memory_aid_list_model = memory_aid_ns.model('MemoryAidList', {
    'memory_aids': fields.List(fields.Nested(memory_aid_output_model))
})

# Conversational QA models
query_input_model = qa_ns.model('QueryInput', {
    'query': fields.String(required=True, description='The question to answer', example='What is the capital of France?'),
    'user_id': fields.String(description='Optional user ID for personalized responses')
})

qa_response_model = qa_ns.model('QAResponse', {
    'answer': fields.String(description='The answer to the question', example='The capital of France is Paris.'),
    'confidence': fields.Float(description='Confidence score for the answer', example=0.95),
    'source': fields.String(description='Source of the information', example='Knowledge base'),
    'query_understood': fields.Boolean(description='Whether the system understood the query', example=True)
})

passage_input_model = qa_ns.model('PassageInput', {
    'text': fields.String(required=True, description='The passage to use for answering questions', 
                        example='Paris is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018.'),
    'title': fields.String(description='Optional title for the passage', example='Paris Information')
})

text_input_model = qa_ns.model('TextInput', {
    'text': fields.String(required=True, description='Text to summarize'),
})

summarize_response_model = qa_ns.model('SummarizeResponse', {
    'summary': fields.String(description='Summarized text'),
    'success': fields.Boolean(description='Success status')
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
        pass

@auth_ns.route('/login')
class UserLogin(Resource):
    @auth_ns.doc('user_login')
    @auth_ns.expect(login_model)
    @auth_ns.response(200, 'Success', token_model)
    @auth_ns.response(401, 'Invalid credentials')
    def post(self):
        """Login user and get authentication token"""
        pass

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
        pass

@user_ns.route('/<user_id>')
@user_ns.param('user_id', 'The user identifier')
class UserResource(Resource):
    @user_ns.doc('get_user', security='apiKey')
    @user_ns.response(200, 'Success', user_model)
    @user_ns.response(401, 'Authentication required')
    @user_ns.response(403, 'Unauthorized')
    @user_ns.response(404, 'User not found')
    def get(self, user_id):
        """Get user details"""
        pass
        
    @user_ns.doc('update_user', security='apiKey')
    @user_ns.expect(update_user_model)
    @user_ns.response(200, 'User updated successfully')
    @user_ns.response(400, 'Validation error')
    @user_ns.response(401, 'Authentication required')
    @user_ns.response(403, 'Unauthorized')
    @user_ns.response(404, 'User not found')
    def put(self, user_id):
        """Update user details"""
        pass
        
    @user_ns.doc('delete_user', security='apiKey')
    @user_ns.response(200, 'User deleted successfully')
    @user_ns.response(401, 'Authentication required')
    @user_ns.response(403, 'Unauthorized')
    @user_ns.response(404, 'User not found')
    def delete(self, user_id):
        """Delete a user"""
        pass

# AI Processing routes
@ai_ns.route('/process')
class AIProcess(Resource):
    @ai_ns.doc('process_ai_request')
    @ai_ns.expect(ai_input_model)
    @ai_ns.response(200, 'Success', ai_response_model)
    @ai_ns.response(400, 'Invalid input')
    @ai_ns.response(500, 'Processing error')
    def post(self):
        """Process a natural language request and categorize responses"""
        pass

@ai_ns.route('/news')
class AINews(Resource):
    @ai_ns.doc('process_news')
    @ai_ns.expect(ai_input_model)
    @ai_ns.response(200, 'Success', category_response_model)
    @ai_ns.response(400, 'Invalid input')
    @ai_ns.response(500, 'Processing error')
    def post(self):
        """Process a news-specific request"""
        pass

@ai_ns.route('/weather')
class AIWeather(Resource):
    @ai_ns.doc('process_weather')
    @ai_ns.expect(ai_input_model)
    @ai_ns.response(200, 'Success', category_response_model)
    @ai_ns.response(400, 'Invalid input')
    @ai_ns.response(500, 'Processing error')
    def post(self):
        """Process a weather-specific request"""
        pass

@ai_ns.route('/news/article')
class NewsArticleAPI(Resource):
    @ai_ns.doc('upload_news_article')
    @ai_ns.expect(api.model('NewsArticleInput', {
        'text': fields.String(required=True, description='The full text of the news article'),
        'title': fields.String(description='Optional title for the article')
    }))
    @ai_ns.response(200, 'Success', api.model('NewsArticleResponse', {
        'success': fields.Boolean(description='Whether the operation was successful'),
        'message': fields.String(description='A message describing the result'),
        'title': fields.String(description='The title of the processed article'),
        'summary': fields.String(description='A summary of the article'),
        'article_length': fields.Integer(description='The length of the processed article'),
        'context_set': fields.Boolean(description='Whether the article was set as context for future queries')
    }))
    @ai_ns.response(400, 'Invalid input')
    @ai_ns.response(500, 'Processing error')
    def post(self):
        """Upload a news article to be used as context for future queries"""
        pass

# Memory Aid routes
@memory_aid_ns.route('')
class MemoryAidList(Resource):
    @memory_aid_ns.doc('list_memory_aids', security='apiKey')
    @memory_aid_ns.response(200, 'Success', [memory_aid_output_model])
    @memory_aid_ns.response(401, 'Authentication required')
    def get(self):
        """Get all memory aids for the current user"""
        pass
        
    @memory_aid_ns.doc('create_memory_aid', security='apiKey')
    @memory_aid_ns.expect(memory_aid_input_model)
    @memory_aid_ns.response(201, 'Memory aid created')
    @memory_aid_ns.response(400, 'Validation error')
    @memory_aid_ns.response(401, 'Authentication required')
    def post(self):
        """Create a new memory aid"""
        pass

@memory_aid_ns.route('/<memory_aid_id>')
@memory_aid_ns.param('memory_aid_id', 'The memory aid identifier')
class MemoryAidResource(Resource):
    @memory_aid_ns.doc('get_memory_aid', security='apiKey')
    @memory_aid_ns.response(200, 'Success', memory_aid_output_model)
    @memory_aid_ns.response(401, 'Authentication required')
    @memory_aid_ns.response(403, 'Unauthorized')
    @memory_aid_ns.response(404, 'Memory aid not found')
    def get(self, memory_aid_id):
        """Get a specific memory aid"""
        pass
        
    @memory_aid_ns.doc('update_memory_aid', security='apiKey')
    @memory_aid_ns.expect(memory_aid_input_model)
    @memory_aid_ns.response(200, 'Memory aid updated')
    @memory_aid_ns.response(400, 'Validation error')
    @memory_aid_ns.response(401, 'Authentication required')
    @memory_aid_ns.response(403, 'Unauthorized')
    @memory_aid_ns.response(404, 'Memory aid not found')
    def put(self, memory_aid_id):
        """Update a memory aid"""
        pass
        
    @memory_aid_ns.doc('delete_memory_aid', security='apiKey')
    @memory_aid_ns.response(200, 'Memory aid deleted')
    @memory_aid_ns.response(401, 'Authentication required')
    @memory_aid_ns.response(403, 'Unauthorized')
    @memory_aid_ns.response(404, 'Memory aid not found')
    def delete(self, memory_aid_id):
        """Delete a memory aid"""
        pass

@memory_aid_ns.route('/search')
class MemoryAidSearch(Resource):
    @memory_aid_ns.doc('search_memory_aids', security='apiKey')
    @memory_aid_ns.expect(memory_aid_search_model)
    @memory_aid_ns.response(200, 'Success', [memory_aid_output_model])
    @memory_aid_ns.response(401, 'Authentication required')
    def post(self):
        """Search for memory aids"""
        pass

# Conversational QA routes
@qa_ns.route('/query')
class QueryAPI(Resource):
    @qa_ns.doc('process_qa_query')
    @qa_ns.expect(query_input_model)
    @qa_ns.response(200, 'Success', qa_response_model)
    @qa_ns.response(400, 'Invalid input')
    @qa_ns.response(500, 'Processing error')
    def post(self):
        """Answer a question using the conversational QA system"""
        pass

@qa_ns.route('/set_passage')
class SetPassageAPI(Resource):
    @qa_ns.doc('set_qa_passage')
    @qa_ns.expect(passage_input_model)
    @qa_ns.response(200, 'Success')
    @qa_ns.response(400, 'Invalid input')
    @qa_ns.response(500, 'Processing error')
    def post(self):
        """Set a passage/context for the QA system to answer questions from"""
        pass

@qa_ns.route('/summarize')
class SummarizeAPI(Resource):
    @qa_ns.doc('summarize_article')
    @qa_ns.expect(text_input_model)
    @qa_ns.response(200, 'Success', summarize_response_model)
    @qa_ns.response(400, 'Invalid input')
    @qa_ns.response(500, 'Processing error')
    def post(self):
        """Summarize a text passage or article"""
        pass

if __name__ == '__main__':
    app.run(debug=True, port=5001) 