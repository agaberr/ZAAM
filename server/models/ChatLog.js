const mongoose = require('mongoose');

const chatLogSchema = new mongoose.Schema({
  user_id: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  conversation: [{
    timestamp: {
      type: Date,
      default: Date.now
    },
    sender: {
      type: String,
      enum: ['user', 'AI'],
      required: true
    },
    message: {
      type: String,
      required: true
    },
    emotion_detected: {
      type: String,
      default: 'neutral'
    },
    response_type: {
      type: String,
      default: 'informational'
    }
  }],
  summary: {
    type: String
  },
  created_at: {
    type: Date,
    default: Date.now
  }
});

const ChatLog = mongoose.model('ChatLog', chatLogSchema);

module.exports = ChatLog;
