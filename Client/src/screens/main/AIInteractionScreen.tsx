import React, { useState, useEffect, useRef } from 'react';
import { View, StyleSheet, ScrollView, KeyboardAvoidingView, Platform, TextInput as RNTextInput } from 'react-native';
import { Text, IconButton, Surface, Avatar, ActivityIndicator, Button } from 'react-native-paper';
import * as Speech from 'expo-speech';
import { useAuth } from '../../context/AuthContext';

// Mock 3D model component (in a real app, you would use a 3D rendering library)
const AI3DModel = () => {
  return (
    <Surface style={styles.modelContainer}>
      <Avatar.Icon size={120} icon="robot" style={styles.avatar} />
      <Text style={styles.modelText}>AI Assistant</Text>
    </Surface>
  );
};

const AIInteractionScreen = () => {
  const { user } = useAuth();
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const scrollViewRef = useRef();
  const inputRef = useRef();

  // Initialize with a welcome message
  useEffect(() => {
    const welcomeMessage = {
      id: 1,
      sender: 'AI',
      text: `Hello ${user?.full_name?.split(' ')[0] || 'there'}! How can I assist you today?`,
      timestamp: new Date(),
    };
    setMessages([welcomeMessage]);
    
    // Speak the welcome message
    speakText(welcomeMessage.text);
  }, []);

  // Auto scroll to bottom when messages change
  useEffect(() => {
    if (scrollViewRef.current) {
      setTimeout(() => {
        scrollViewRef.current.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [messages]);

  const handleSend = async () => {
    if (!inputText.trim()) return;
    
    // Add user message
    const userMessage = {
      id: messages.length + 1,
      sender: 'user',
      text: inputText,
      timestamp: new Date(),
    };
    
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInputText('');
    setIsProcessing(true);
    
    try {
      // In a real app, you would send the message to your AI backend
      // const response = await aiService.sendMessage(inputText);
      
      // Simulate AI processing delay
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Mock AI response based on user input
      let aiResponse = '';
      
      if (inputText.toLowerCase().includes('hello') || inputText.toLowerCase().includes('hi')) {
        aiResponse = `Hello again! How are you feeling today?`;
      } else if (inputText.toLowerCase().includes('medication') || inputText.toLowerCase().includes('medicine')) {
        aiResponse = `Your next medication is Aspirin at 12:00 PM. Would you like me to remind you?`;
      } else if (inputText.toLowerCase().includes('appointment') || inputText.toLowerCase().includes('doctor')) {
        aiResponse = `You have a doctor's appointment tomorrow at 3:00 PM with Dr. Smith. Would you like me to add it to your calendar?`;
      } else if (inputText.toLowerCase().includes('family') || inputText.toLowerCase().includes('daughter')) {
        aiResponse = `Your daughter Jane called yesterday. Would you like me to call her for you?`;
      } else if (inputText.toLowerCase().includes('time') || inputText.toLowerCase().includes('date')) {
        const now = new Date();
        aiResponse = `It's currently ${now.toLocaleTimeString()} on ${now.toLocaleDateString()}.`;
      } else if (inputText.toLowerCase().includes('weather')) {
        aiResponse = `The weather today is sunny with a high of 75°F. It's a beautiful day!`;
      } else if (inputText.toLowerCase().includes('remind') || inputText.toLowerCase().includes('reminder')) {
        aiResponse = `I've set a reminder for you. I'll notify you at the appropriate time.`;
      } else {
        aiResponse = `I understand you're saying something about "${inputText}". How can I help you with that?`;
      }
      
      // Add AI response
      const aiMessage = {
        id: messages.length + 2,
        sender: 'AI',
        text: aiResponse,
        timestamp: new Date(),
      };
      
      setMessages(prevMessages => [...prevMessages, aiMessage]);
      
      // Speak the AI response
      speakText(aiResponse);
    } catch (error) {
      console.error('Error processing message', error);
      
      // Add error message
      const errorMessage = {
        id: messages.length + 2,
        sender: 'AI',
        text: 'I'm sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
        isError: true,
      };
      
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    } finally {
      setIsProcessing(false);
    }
  };

  const startListening = () => {
    // In a real app, you would implement speech recognition
    // For this demo, we'll simulate it
    setIsListening(true);
    
    // Simulate speech recognition
    setTimeout(() => {
      setIsListening(false);
      setInputText('What medications do I need to take today?');
    }, 2000);
  };

  const speakText = (text) => {
    setIsSpeaking(true);
    
    Speech.speak(text, {
      language: 'en',
      pitch: 1.0,
      rate: 0.9,
      onDone: () => setIsSpeaking(false),
      onError: () => setIsSpeaking(false),
    });
  };

  const stopSpeaking = () => {
    Speech.stop();
    setIsSpeaking(false);
  };

  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <KeyboardAvoidingView 
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
    >
      <View style={styles.modelWrapper}>
        <AI3DModel />
        {isSpeaking && (
          <View style={styles.speakingIndicator}>
            <Text style={styles.speakingText}>Speaking...</Text>
            <Button 
              mode="text" 
              compact 
              onPress={stopSpeaking}
              style={styles.stopButton}
            >
              Stop
            </Button>
          </View>
        )}
      </View>
      
      <View style={styles.chatContainer}>
        <ScrollView 
          ref={scrollViewRef}
          style={styles.messagesContainer}
          contentContainerStyle={styles.messagesContent}
        >
          {messages.map((message) => (
            <View 
              key={message.id} 
              style={[
                styles.messageBubble,
                message.sender === 'user' ? styles.userBubble : styles.aiBubble,
                message.isError && styles.errorBubble,
              ]}
            >
              <Text style={styles.messageText}>{message.text}</Text>
              <Text style={styles.timestamp}>{formatTime(message.timestamp)}</Text>
            </View>
          ))}
          
          {isProcessing && (
            <View style={[styles.messageBubble, styles.aiBubble]}>
              <View style={styles.typingIndicator}>
                <ActivityIndicator size={20} color="#6200ee" />
                <Text style={styles.typingText}>AI is thinking...</Text>
              </View>
            </View>
          )}
        </ScrollView>
        
        <View style={styles.inputContainer}>
          <RNTextInput
            ref={inputRef}
            style={styles.input}
            value={inputText}
            onChangeText={setInputText}
            placeholder="Type a message..."
            multiline
          />
          
          <IconButton
            icon="microphone"
            size={24}
            color={isListening ? '#6200ee' : '#757575'}
            style={styles.micButton}
            onPress={startListening}
            disabled={isProcessing}
          />
          
          <IconButton
            icon="send"
            size={24}
            color="#6200ee"
            style={styles.sendButton}
            onPress={handleSend}
            disabled={!inputText.trim() || isProcessing}
          />
        </View>
      </View>
      
      <View style={styles.castContainer}>
        <Button 
          mode="outlined" 
          icon="cast" 
          onPress={() => {/* Implement TV casting */}}
          style={styles.castButton}
        >
          Cast to TV
        </Button>
      </View>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  modelWrapper: {
    alignItems: 'center',
    paddingTop: 20,
    paddingBottom: 10,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  modelContainer: {
    width: 150,
    height: 150,
    borderRadius: 75,
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 4,
  },
  avatar: {
    backgroundColor: '#6200ee',
  },
  modelText: {
    marginTop: 10,
    fontSize: 16,
    fontWeight: 'bold',
  },
  speakingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 5,
  },
  speakingText: {
    color: '#6200ee',
    fontStyle: 'italic',
  },
  stopButton: {
    marginLeft: 5,
  },
  chatContainer: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  messagesContainer: {
    flex: 1,
    padding: 10,
  },
  messagesContent: {
    paddingBottom: 10,
  },
  messageBubble: {
    maxWidth: '80%',
    padding: 12,
    borderRadius: 20,
    marginBottom: 10,
  },
  userBubble: {
    backgroundColor: '#e1f5fe',
    alignSelf: 'flex-end',
    borderBottomRightRadius: 5,
  },
  aiBubble: {
    backgroundColor: '#f0f0f0',
    alignSelf: 'flex-start',
    borderBottomLeftRadius: 5,
  },
  errorBubble: {
    backgroundColor: '#ffebee',
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22,
  },
  timestamp: {
    fontSize: 12,
    color: '#757575',
    alignSelf: 'flex-end',
    marginTop: 5,
  },
  typingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  typingText: {
    marginLeft: 10,
    fontStyle: 'italic',
    color: '#757575',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  input: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 10,
    maxHeight: 100,
    fontSize: 16,
  },
  micButton: {
    marginHorizontal: 5,
  },
  sendButton: {
    marginLeft: 5,
  },
  castContainer: {
    padding: 10,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    alignItems: 'center',
  },
  castButton: {
    width: '100%',
  },
});

export default AIInteractionScreen;
