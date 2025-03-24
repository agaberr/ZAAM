import React, { useState, useRef, useEffect } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, Image, Animated, Modal, ScrollView, Alert, KeyboardAvoidingView, Platform } from 'react-native';
import { Ionicons, MaterialCommunityIcons, FontAwesome5 } from '@expo/vector-icons';
import { Button, Surface, ActivityIndicator, Chip, Avatar, TextInput, IconButton } from 'react-native-paper';
import { Audio } from 'expo-av';
import * as Speech from 'expo-speech';
import { useAuth } from '../context/AuthContext';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: Date;
}

export default function TalkToAIScreen({ setActiveTab }) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: "Hello! I'm your AI assistant. How can I help you today? You can ask me about your medication, set reminders, or just chat.",
      sender: 'ai',
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [showControls, setShowControls] = useState(true);
  const [castModalVisible, setCastModalVisible] = useState(false);
  const [castingActive, setCastingActive] = useState(false);
  const [castDevice, setCastDevice] = useState('');
  const [moodModalVisible, setMoodModalVisible] = useState(false);
  const [aiMood, setAiMood] = useState('friendly');
  const [showTips, setShowTips] = useState(false);
  const scrollViewRef = useRef<ScrollView>(null);
  const { createCalendarEvent, googleCredentials, processNaturalLanguageReminder } = useAuth();

  // Animation values
  const pulseAnim = useRef(new Animated.Value(1)).current;

  // Sample devices for casting
  const castDevices = [
    { id: '1', name: 'Living Room TV', type: 'Samsung Smart TV' },
    { id: '2', name: 'Bedroom TV', type: 'LG WebOS TV' },
    { id: '3', name: 'Kitchen Display', type: 'Google Nest Hub' }
  ];

  // Sample quick phrases
  const quickPhrases = [
    "What day is it today?",
    "Remind me to take my medicine",
    "What's on my schedule?",
    "Call my caregiver",
    "Tell me about my family",
    "Help me remember where I put my glasses"
  ];

  // Request audio permissions
  useEffect(() => {
    const getPermissions = async () => {
      const { status } = await Audio.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Please allow microphone access to use voice features.');
      }
    };

    getPermissions();
  }, []);

  // Detect reminder creation intents
  const detectReminderIntent = (text: string) => {
    const lowerText = text.toLowerCase();
    
    // Check if this is a reminder creation request
    if (
      lowerText.includes('remind me') || 
      lowerText.includes('set a reminder') || 
      lowerText.includes('create a reminder') ||
      lowerText.includes('add reminder') ||
      lowerText.includes('set reminder')
    ) {
      return true;
    }
    
    return false;
  };
  
  // Extract reminder details from text
  const extractReminderDetails = (text: string) => {
    const lowerText = text.toLowerCase();
    
    // Default values
    let title = "New Reminder";
    let description = "";
    let dateTime = new Date();
    dateTime.setMinutes(dateTime.getMinutes() + 30); // Default to 30 minutes from now
    
    // Extract title (text after "to" or "about" or after reminder keywords)
    const titleMatches = text.match(/remind me (?:to|about) (.*?)(?:on|at|tomorrow|next|$)/i) || 
                         text.match(/set a reminder (?:to|for|about) (.*?)(?:on|at|tomorrow|next|$)/i) ||
                         text.match(/create a reminder (?:to|for|about) (.*?)(?:on|at|tomorrow|next|$)/i);
    
    if (titleMatches && titleMatches[1]) {
      title = titleMatches[1].trim();
      description = "Voice reminder: " + title;
    }
    
    // Extract time
    if (lowerText.includes('at ')) {
      const timeMatches = lowerText.match(/at (\d+)(?::(\d+))?\s*(am|pm)?/i);
      if (timeMatches) {
        let hours = parseInt(timeMatches[1]);
        const minutes = timeMatches[2] ? parseInt(timeMatches[2]) : 0;
        const period = timeMatches[3]?.toLowerCase();
        
        // Adjust hours for PM
        if (period === 'pm' && hours < 12) {
          hours += 12;
        } else if (period === 'am' && hours === 12) {
          hours = 0;
        }
        
        dateTime.setHours(hours, minutes, 0, 0);
      }
    }
    
    // Extract date
    if (lowerText.includes('tomorrow')) {
      dateTime.setDate(dateTime.getDate() + 1);
      dateTime.setHours(9, 0, 0, 0); // Default to 9 AM if only day is specified
    } else if (lowerText.includes('next week')) {
      dateTime.setDate(dateTime.getDate() + 7);
      dateTime.setHours(9, 0, 0, 0);
    } else if (lowerText.match(/on (monday|tuesday|wednesday|thursday|friday|saturday|sunday)/i)) {
      const dayMatch = lowerText.match(/on (monday|tuesday|wednesday|thursday|friday|saturday|sunday)/i);
      if (dayMatch) {
        const dayOfWeek = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']
          .indexOf(dayMatch[1].toLowerCase());
        
        const today = dateTime.getDay();
        let daysToAdd = (dayOfWeek - today) % 7;
        if (daysToAdd <= 0) daysToAdd += 7; // Next week if day has passed
        
        dateTime.setDate(dateTime.getDate() + daysToAdd);
        dateTime.setHours(9, 0, 0, 0);
      }
    }
    
    return { title, description, dateTime };
  };
  
  // Create a reminder in Google Calendar
  const createReminderFromVoice = async (text: string) => {
    try {
      if (!googleCredentials) {
        const aiResponse = "I need access to your Google Calendar to set reminders. Please sign in with Google in the Reminders tab.";
        addMessage(aiResponse, 'ai');
        speakMessage(aiResponse);
        return;
      }
      
      // Extract details
      const { title, description, dateTime } = extractReminderDetails(text);
      
      // Create event
      await createCalendarEvent(title, description, dateTime);
      
      // Format confirmation message
      const timeString = dateTime.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
      const dateString = dateTime.toLocaleDateString([], {weekday: 'long', month: 'long', day: 'numeric'});
      const confirmationMsg = `I've set a reminder for "${title}" on ${dateString} at ${timeString}.`;
      
      // Add AI response
      addMessage(confirmationMsg, 'ai');
      speakMessage(confirmationMsg);
      
    } catch (error) {
      console.error('Failed to create reminder:', error);
      const errorMsg = "I'm sorry, I couldn't create your reminder. Please try again.";
      addMessage(errorMsg, 'ai');
      speakMessage(errorMsg);
    }
  };

  const startRecording = async () => {
    try {
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });
      
      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      
      setRecording(recording);
      setIsRecording(true);
    } catch (err) {
      console.error('Failed to start recording', err);
      Alert.alert('Error', 'Failed to start recording. Please try again.');
    }
  };

  const stopRecording = async () => {
    if (!recording) return;
    
    setIsRecording(false);
    setIsProcessing(true);
    
    try {
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      setRecording(null);
      
      if (uri) {
        // In a real app, you would send this audio file to a speech-to-text service
        // For this demo, we'll simulate a response after a short delay
        setTimeout(() => {
          // For testing, simulate receiving transcribed text
          const simulatedText = "Remind me to take my medication tomorrow at 9 AM";
          setInputText(simulatedText);
          handleSendMessage(simulatedText);
        }, 1500);
      }
    } catch (err) {
      console.error('Failed to stop recording', err);
      setIsProcessing(false);
      const errorMsg = "I'm sorry, there was a problem with the recording. Please try again.";
      addMessage(errorMsg, 'ai');
    }
  };

  const handleSendMessage = async (text: string = inputText) => {
    if (!text.trim()) return;
    
    // Add user message
    addMessage(text, 'user');
    setInputText('');
    setIsProcessing(true);
    
    try {
      // Check if this is a reminder request
      if (detectReminderIntent(text)) {
        // Use the NLP processor to handle this reminder request
        const result = await processNaturalLanguageReminder(text);
        
        // Get the intent and slots
        const { intent, slots } = result;
        
        if (intent === 'create_event' && slots.action) {
          const timeStr = slots.time || 'now';
          const confirmationMsg = `I've set a reminder for "${slots.action}" at ${timeStr}.`;
          addMessage(confirmationMsg, 'ai');
          speakMessage(confirmationMsg);
        } else {
          const response = "I'm not sure what kind of reminder you want to set. Please try again with more details.";
          addMessage(response, 'ai');
          speakMessage(response);
        }
      } else {
        // Handle other types of messages normally
        processUserMessage(text);
      }
    } catch (error) {
      console.error('Error processing message:', error);
      const errorMsg = "I'm sorry, I couldn't process your request. Please try again.";
      addMessage(errorMsg, 'ai');
      speakMessage(errorMsg);
    } finally {
      setIsProcessing(false);
    }
  };

  const processUserMessage = async (text: string) => {
    // Handle other types of queries (simplified for this demo)
    let response = '';
    
    const lowerText = text.toLowerCase();
    if (lowerText.includes('hello') || lowerText.includes('hi')) {
      response = 'Hello! How can I help you today?';
    } else if (lowerText.includes('how are you')) {
      response = "I'm just a digital assistant, but I'm functioning well! How can I assist you?";
    } else if (lowerText.includes('medication') || lowerText.includes('medicine') || lowerText.includes('pill')) {
      response = 'Would you like me to remind you about your medication? I can set up a calendar reminder for you.';
    } else if (lowerText.includes('reminder') || lowerText.includes('calendar')) {
      response = 'I can help you set reminders. Just say something like "Remind me to take my medicine at 9 AM" or "Set a reminder for my doctor appointment on Friday at 3 PM."';
    } else if (lowerText.includes('thank you') || lowerText.includes('thanks')) {
      response = "You're welcome! I'm here to help anytime.";
    } else {
      response = "I'm not sure how to respond to that. I can help you set reminders or answer questions about using the app. Just let me know what you need.";
    }
    
    // Add AI response with a slight delay to appear more natural
    setTimeout(() => {
      addMessage(response, 'ai');
      speakMessage(response);
    }, 500);
  };

  const addMessage = (text: string, sender: 'user' | 'ai') => {
    const newMessage: Message = {
      id: Date.now().toString(),
      text,
      sender,
      timestamp: new Date(),
    };
    
    setMessages(prevMessages => [...prevMessages, newMessage]);
    
    // Scroll to bottom with a slight delay to ensure the message is rendered
    setTimeout(() => {
      scrollViewRef.current?.scrollToEnd({ animated: true });
    }, 100);
  };

  const speakMessage = async (text: string) => {
    if (isSpeaking) {
      Speech.stop();
    }
    
    setIsSpeaking(true);
    
    try {
      await Speech.speak(text, {
        language: 'en',
        rate: 0.9,
        onDone: () => setIsSpeaking(false),
        onError: () => setIsSpeaking(false),
      });
    } catch (error) {
      console.error('Speech error:', error);
      setIsSpeaking(false);
    }
  };

  const handleCastPress = () => {
    setCastModalVisible(true);
  };

  const startCasting = (deviceId, deviceName) => {
    setCastingActive(true);
    setCastDevice(deviceName);
    setCastModalVisible(false);
  };

  const stopCasting = () => {
    setCastingActive(false);
    setCastDevice('');
  };

  const changeMood = (mood) => {
    setAiMood(mood);
    setMoodModalVisible(false);
  };

  return (
    <KeyboardAvoidingView 
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={80}
    >
      <View style={styles.header}>
        <Avatar.Icon size={40} icon="robot" style={styles.avatar} />
        <View>
          <Text style={styles.headerTitle}>AI Assistant</Text>
          <Text style={styles.headerSubtitle}>Your personal health companion</Text>
        </View>
      </View>
      
      <ScrollView
        style={styles.messagesContainer}
        ref={scrollViewRef}
        contentContainerStyle={styles.messagesContent}
      >
        {messages.map((message) => (
          <View
            key={message.id}
            style={[
              styles.messageBubble,
              message.sender === 'user' ? styles.userBubble : styles.aiBubble,
            ]}
          >
            <Text style={styles.messageText}>{message.text}</Text>
            <Text style={styles.timestampText}>
              {message.timestamp.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
            </Text>
          </View>
        ))}
        
        {isProcessing && (
          <View style={styles.processingContainer}>
            <ActivityIndicator size="small" color="#4285F4" />
            <Text style={styles.processingText}>Processing your request...</Text>
          </View>
        )}
      </ScrollView>
      
      <Surface style={styles.inputContainer} elevation={4}>
        <View style={styles.suggestedPrompts}>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            <Chip 
              style={styles.promptChip} 
              onPress={() => setInputText('Remind me to take my medication at 9 AM')}>
              Medication reminder
            </Chip>
            <Chip 
              style={styles.promptChip} 
              onPress={() => setInputText('Set a reminder for my doctor appointment next Monday')}>
              Doctor appointment
            </Chip>
            <Chip 
              style={styles.promptChip} 
              onPress={() => setInputText('Remind me to drink water every 2 hours')}>
              Hydration reminder
            </Chip>
            <Chip 
              style={styles.promptChip} 
              onPress={() => setInputText('How do I use the memory aids?')}>
              Help with app
            </Chip>
          </ScrollView>
        </View>
        
        <View style={styles.inputRow}>
          <TextInput
            mode="outlined"
            placeholder="Type a message..."
            value={inputText}
            onChangeText={setInputText}
            style={styles.textInput}
            right={
              inputText ? 
              <TextInput.Icon icon="close" onPress={() => setInputText('')} /> : 
              undefined
            }
          />
          
          {inputText ? (
            <IconButton
              icon="send"
              size={24}
              onPress={() => handleSendMessage()}
              style={styles.sendButton}
            />
          ) : (
            <TouchableOpacity
              style={styles.micButton}
              onPress={isRecording ? stopRecording : startRecording}
            >
              <FontAwesome5
                name={isRecording ? "stop" : "microphone"}
                size={20}
                color="#FFF"
              />
            </TouchableOpacity>
          )}
        </View>
      </Surface>

      {/* Cast Modal */}
      <Modal
        visible={castModalVisible}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setCastModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Cast to Device</Text>
            <View style={styles.castDevicesList}>
              {castDevices.map(device => (
                <TouchableOpacity
                  key={device.id}
                  style={styles.castDeviceItem}
                  onPress={() => startCasting(device.id, device.name)}
                >
                  <Ionicons name="tv" size={24} color="#4285F4" style={styles.castDeviceIcon} />
                  <View style={styles.castDeviceInfo}>
                    <Text style={styles.castDeviceName}>{device.name}</Text>
                    <Text style={styles.castDeviceType}>{device.type}</Text>
                  </View>
                  <Ionicons name="chevron-forward" size={20} color="#777" />
                </TouchableOpacity>
              ))}
            </View>
            <Button 
              mode="text" 
              onPress={() => setCastModalVisible(false)}
              style={styles.cancelButton}
            >
              Cancel
            </Button>
          </View>
        </View>
      </Modal>

      {/* Mood Selection Modal */}
      <Modal
        visible={moodModalVisible}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setMoodModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Change AI Personality</Text>
            <View style={styles.moodOptionsList}>
              <TouchableOpacity
                style={[styles.moodOption, aiMood === 'friendly' ? styles.selectedMood : {}]}
                onPress={() => changeMood('friendly')}
              >
                <Ionicons name="happy" size={24} color="#4285F4" />
                <Text style={styles.moodOptionText}>Friendly</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.moodOption, aiMood === 'professional' ? styles.selectedMood : {}]}
                onPress={() => changeMood('professional')}
              >
                <Ionicons name="briefcase" size={24} color="#4285F4" />
                <Text style={styles.moodOptionText}>Professional</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.moodOption, aiMood === 'calm' ? styles.selectedMood : {}]}
                onPress={() => changeMood('calm')}
              >
                <Ionicons name="water" size={24} color="#4285F4" />
                <Text style={styles.moodOptionText}>Calm</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.moodOption, aiMood === 'cheerful' ? styles.selectedMood : {}]}
                onPress={() => changeMood('cheerful')}
              >
                <Ionicons name="sunny" size={24} color="#4285F4" />
                <Text style={styles.moodOptionText}>Cheerful</Text>
              </TouchableOpacity>
            </View>
            <Button 
              mode="text" 
              onPress={() => setMoodModalVisible(false)}
              style={styles.cancelButton}
            >
              Cancel
            </Button>
          </View>
        </View>
      </Modal>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5F7FA',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
  },
  avatar: {
    backgroundColor: '#4285F4',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#777',
  },
  messagesContainer: {
    flex: 1,
    width: '100%',
  },
  messagesContent: {
    padding: 10,
    gap: 10,
  },
  messageBubble: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    borderRadius: 10,
    backgroundColor: 'white',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  userBubble: {
    backgroundColor: '#E8F1FF',
  },
  aiBubble: {
    backgroundColor: '#F5F5F5',
  },
  messageText: {
    flex: 1,
    fontSize: 16,
    lineHeight: 22,
  },
  timestampText: {
    fontSize: 12,
    color: '#777',
  },
  inputContainer: {
    width: '100%',
    padding: 10,
  },
  suggestedPrompts: {
    marginBottom: 10,
  },
  promptChip: {
    backgroundColor: '#F5F5F5',
    padding: 10,
    borderRadius: 10,
    marginRight: 10,
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  textInput: {
    flex: 1,
  },
  sendButton: {
    marginLeft: 10,
  },
  micButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#4285F4',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: 15,
    padding: 20,
    width: '80%',
    maxWidth: 400,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
    textAlign: 'center',
  },
  castDevicesList: {
    gap: 10,
    marginBottom: 15,
  },
  castDeviceItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F5F5F5',
    padding: 12,
    borderRadius: 10,
  },
  castDeviceIcon: {
    marginRight: 15,
  },
  castDeviceInfo: {
    flex: 1,
  },
  castDeviceName: {
    fontSize: 16,
    fontWeight: '500',
  },
  castDeviceType: {
    fontSize: 14,
    color: '#777',
  },
  moodOptionsList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    gap: 10,
    marginBottom: 15,
  },
  moodOption: {
    width: '48%',
    backgroundColor: '#F5F5F5',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
  },
  selectedMood: {
    backgroundColor: '#E8F1FF',
    borderWidth: 2,
    borderColor: '#4285F4',
  },
  moodOptionText: {
    marginTop: 8,
    fontSize: 14,
    fontWeight: '500',
  },
  cancelButton: {
    alignSelf: 'center',
  },
  processingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
  },
  processingText: {
    marginLeft: 10,
    fontSize: 16,
    fontWeight: 'bold',
  },
}); 