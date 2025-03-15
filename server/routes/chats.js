const express = require('express');
const ChatLog = require('../models/ChatLog');
const auth = require('../middleware/auth');
const router = express.Router();

// Get chat history
router.get('/history', auth, async (req, res) => {
  try {
    const chatLogs = await ChatLog.find({ user_id: req.userId })
      .sort({ created_at: -1 })
      .limit(10);
    res.json(chatLogs);
  } catch (error) {
    console.error('Error fetching chat history:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Add new chat message
router.post('/message', auth, async (req, res) => {
  try {
    const { message } = req.body;
    
    // Create new chat log or append to existing conversation
    let chatLog = await ChatLog.findOne({ 
      user_id: req.userId,
      created_at: { 
        $gte: new Date(new Date().setHours(0, 0, 0, 0)) 
      }
    });
    
    if (!chatLog) {
      chatLog = new ChatLog({
        user_id: req.userId,
        conversation: []
      });
    }
    
    // Add user message
    chatLog.conversation.push({
      sender: 'user',
      message,
      timestamp: new Date(),
      emotion_detected: 'analyzing'
    });
    
    // Generate AI response (mock implementation)
    const aiResponse = {
      sender: 'AI',
      message: `I understand you're saying: "${message}". How can I help you with that?`,
      timestamp: new Date(),
      response_type: 'informational'
    };
    
    chatLog.conversation.push(aiResponse);
    
    // Update summary
    chatLog.summary = `Conversation about: ${message.substring(0, 50)}...`;
    
    await chatLog.save();
    res.json(aiResponse);
  } catch (error) {
    console.error('Error processing chat message:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router;
