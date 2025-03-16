import React, { useState, useRef, useEffect } from 'react';
import { 
  StyleSheet, 
  View, 
  Image, 
  TouchableOpacity, 
  Dimensions, 
  Animated, 
  ScrollView,
  Modal
} from 'react-native';
import { 
  Text, 
  IconButton, 
  Button, 
  Surface, 
  ActivityIndicator,
  Portal,
  Dialog,
  Chip
} from 'react-native-paper';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons, MaterialCommunityIcons, FontAwesome5 } from '@expo/vector-icons';
import * as Speech from 'expo-speech';

const { width, height } = Dimensions.get('window');

export default function TalkToAIScreen() {
  const router = useRouter();
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [userMessage, setUserMessage] = useState('');
  const [aiResponse, setAiResponse] = useState('');
  const [showControls, setShowControls] = useState(true);
  const [castModalVisible, setCastModalVisible] = useState(false);
  const [castingActive, setCastingActive] = useState(false);
  const [castDevice, setCastDevice] = useState('');
  const [moodModalVisible, setMoodModalVisible] = useState(false);
  const [aiMood, setAiMood] = useState('friendly');
  const [showTips, setShowTips] = useState(false);
  const [conversationHistory, setConversationHistory] = useState([
    { sender: 'ai', message: 'Hello Amit! How can I help you today?' }
  ]);

  // Animation values
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const fadeAnim = useRef(new Animated.Value(1)).current;

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

  useEffect(() => {
    // Pulse animation for the avatar when speaking
    if (isSpeaking) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.05,
            duration: 1000,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 1000,
            useNativeDriver: true,
          })
        ])
      ).start();
    } else {
      pulseAnim.setValue(1);
    }
  }, [isSpeaking]);

  useEffect(() => {
    // Fade controls after inactivity
    const timer = setTimeout(() => {
      if (!isListening && !isSpeaking) {
        Animated.timing(fadeAnim, {
          toValue: 0,
          duration: 2000,
          useNativeDriver: true,
        }).start(() => setShowControls(false));
      }
    }, 5000);

    return () => clearTimeout(timer);
  }, [isListening, isSpeaking]);

  const handleScreenTouch = () => {
    setShowControls(true);
    fadeAnim.setValue(1);
  };

  const startListening = () => {
    setIsListening(true);
    // Simulate speech recognition
    setTimeout(() => {
      const randomPhrase = quickPhrases[Math.floor(Math.random() * quickPhrases.length)];
      setUserMessage(randomPhrase);
      setIsListening(false);
      processUserInput(randomPhrase);
    }, 2000);
  };

  const processUserInput = (input: string) => {
    // Add user message to conversation
    setConversationHistory(prev => [...prev, { sender: 'user', message: input }]);
    
    // Simulate AI thinking
    setTimeout(() => {
      let response = '';
      
      // Simple response logic based on input
      if (input.toLowerCase().includes('day')) {
        const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
        const today = days[new Date().getDay()];
        response = `Today is ${today}, ${new Date().toLocaleDateString()}.`;
      } 
      else if (input.toLowerCase().includes('medicine') || input.toLowerCase().includes('medication')) {
        response = "You need to take your Aspirin at 9:00 AM and Donepezil at 1:00 PM. Would you like me to set a reminder?";
      }
      else if (input.toLowerCase().includes('schedule')) {
        response = "You have a doctor's appointment at 11:30 AM today, and memory exercises scheduled for 3:00 PM.";
      }
      else if (input.toLowerCase().includes('call')) {
        response = "I'll call your caregiver Jane for you. Connecting now...";
      }
      else if (input.toLowerCase().includes('family')) {
        response = "Your daughter Sarah is coming to visit this weekend. Would you like to see some family photos?";
      }
      else if (input.toLowerCase().includes('glasses')) {
        response = "According to your last interaction, you left your glasses on the coffee table in the living room.";
      }
      else {
        response = "I'm here to help you. Would you like to know about your medications, schedule, or would you like me to call someone for you?";
      }
      
      setAiResponse(response);
      speakResponse(response);
      
      // Add AI response to conversation
      setConversationHistory(prev => [...prev, { sender: 'ai', message: response }]);
    }, 1500);
  };

  const speakResponse = (text: string) => {
    setIsSpeaking(true);
    
    // Use expo-speech to speak the response
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

  const handleCastPress = () => {
    setCastModalVisible(true);
  };

  const startCasting = (deviceId: string, deviceName: string) => {
    setCastingActive(true);
    setCastDevice(deviceName);
    setCastModalVisible(false);
    
    // Show a toast or notification that casting has started
    // This would be implemented with actual casting functionality
  };

  const stopCasting = () => {
    setCastingActive(false);
    setCastDevice('');
  };

  const changeMood = (mood: string) => {
    setAiMood(mood);
    setMoodModalVisible(false);
    
    // This would change the AI's speaking style and potentially the avatar appearance
    let welcomeMessage = '';
    
    switch(mood) {
      case 'friendly':
        welcomeMessage = "I've switched to a friendly tone. How can I help you today?";
        break;
      case 'professional':
        welcomeMessage = "I've adjusted to a more professional demeanor. How may I assist you?";
        break;
      case 'calm':
        welcomeMessage = "I'm now speaking in a calming voice. What would you like to talk about?";
        break;
      case 'cheerful':
        welcomeMessage = "I'm feeling cheerful now! What can I do to brighten your day?";
        break;
    }
    
    setConversationHistory(prev => [...prev, { sender: 'ai', message: welcomeMessage }]);
    speakResponse(welcomeMessage);
  };

  const renderAvatarBasedOnMood = () => {
    // In a real implementation, you would have different 3D models or animations
    // For now, we'll just use the same image with a note about the mood
    return (
      <View style={styles.avatarContainer}>
        <Animated.View 
          style={[
            styles.avatarWrapper,
            { transform: [{ scale: pulseAnim }] }
          ]}
        >
          <Image 
            source={require('./assets/ai-avatar.png')} 
            style={styles.avatarImage} 
            resizeMode="contain"
          />
          {isSpeaking && (
            <View style={styles.speakingIndicator}>
              <View style={[styles.soundWave, styles.wave1]} />
              <View style={[styles.soundWave, styles.wave2]} />
              <View style={[styles.soundWave, styles.wave3]} />
            </View>
          )}
        </Animated.View>
        <Chip 
          style={styles.moodChip} 
          icon={() => {
            switch(aiMood) {
              case 'friendly': return <Ionicons name="happy" size={16} color="#4285F4" />;
              case 'professional': return <Ionicons name="briefcase" size={16} color="#4285F4" />;
              case 'calm': return <Ionicons name="water" size={16} color="#4285F4" />;
              case 'cheerful': return <Ionicons name="sunny" size={16} color="#4285F4" />;
              default: return <Ionicons name="happy" size={16} color="#4285F4" />;
            }
          }}
        >
          {aiMood.charAt(0).toUpperCase() + aiMood.slice(1)} Mode
        </Chip>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <TouchableOpacity 
        activeOpacity={1} 
        style={styles.fullScreenTouch}
        onPress={handleScreenTouch}
      >
        {/* 3D AI Avatar */}
        {renderAvatarBasedOnMood()}
        
        {/* Conversation bubble for latest AI response */}
        {aiResponse && (
          <View style={styles.responseBubble}>
            <Text style={styles.responseText}>{aiResponse}</Text>
          </View>
        )}
        
        {/* Controls overlay */}
        {showControls && (
          <Animated.View 
            style={[
              styles.controlsOverlay,
              { opacity: fadeAnim }
            ]}
          >
            {/* Header with back button and options */}
            <View style={styles.header}>
              <IconButton
                icon="arrow-left"
                size={24}
                onPress={() => router.back()}
                style={styles.headerButton}
              />
              <View style={styles.headerActions}>
                {castingActive ? (
                  <Chip 
                    icon="cast-connected" 
                    onPress={stopCasting}
                    style={styles.castingChip}
                  >
                    Casting to {castDevice}
                  </Chip>
                ) : (
                  <IconButton
                    icon="cast"
                    size={24}
                    onPress={handleCastPress}
                    style={styles.headerButton}
                  />
                )}
                <IconButton
                  icon="dots-vertical"
                  size={24}
                  onPress={() => setMoodModalVisible(true)}
                  style={styles.headerButton}
                />
              </View>
            </View>
            
            {/* Microphone button and status */}
            <View style={styles.microphoneContainer}>
              <TouchableOpacity 
                style={[
                  styles.microphoneButton,
                  isListening ? styles.listeningButton : {}
                ]}
                onPress={isListening ? () => setIsListening(false) : startListening}
                disabled={isSpeaking}
              >
                {isListening ? (
                  <ActivityIndicator size={30} color="white" />
                ) : (
                  <Ionicons name="mic" size={30} color="white" />
                )}
              </TouchableOpacity>
              <Text style={styles.microphoneStatus}>
                {isListening ? 'Listening...' : 'Tap to speak'}
              </Text>
            </View>
            
            {/* Quick phrases */}
            <ScrollView 
              horizontal 
              showsHorizontalScrollIndicator={false}
              style={styles.quickPhrasesContainer}
              contentContainerStyle={styles.quickPhrasesContent}
            >
              {quickPhrases.map((phrase, index) => (
                <TouchableOpacity 
                  key={index}
                  style={styles.quickPhraseButton}
                  onPress={() => {
                    setUserMessage(phrase);
                    processUserInput(phrase);
                  }}
                  disabled={isListening || isSpeaking}
                >
                  <Text style={styles.quickPhraseText}>{phrase}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
            
            {/* Bottom action buttons */}
            <View style={styles.bottomActions}>
              <IconButton
                icon="history"
                size={24}
                onPress={() => router.push('/dashboard/activity')}
                style={styles.actionButton}
              />
              <IconButton
                icon={isSpeaking ? "stop" : "volume-high"}
                size={24}
                onPress={isSpeaking ? stopSpeaking : () => speakResponse(aiResponse)}
                style={styles.actionButton}
                disabled={!aiResponse}
              />
              <IconButton
                icon="lightbulb-on"
                size={24}
                onPress={() => setShowTips(!showTips)}
                style={styles.actionButton}
              />
            </View>
          </Animated.View>
        )}
        
        {/* Tips panel */}
        {showTips && (
          <Surface style={styles.tipsPanel}>
            <View style={styles.tipsPanelHeader}>
              <Text style={styles.tipsPanelTitle}>Tips for talking to your AI</Text>
              <IconButton
                icon="close"
                size={20}
                onPress={() => setShowTips(false)}
              />
            </View>
            <ScrollView style={styles.tipsScrollView}>
              <View style={styles.tipItem}>
                <Ionicons name="calendar" size={24} color="#4285F4" style={styles.tipIcon} />
                <View style={styles.tipContent}>
                  <Text style={styles.tipTitle}>Ask about your schedule</Text>
                  <Text style={styles.tipDescription}>Try "What's on my calendar today?"</Text>
                </View>
              </View>
              <View style={styles.tipItem}>
                <Ionicons name="medical" size={24} color="#4285F4" style={styles.tipIcon} />
                <View style={styles.tipContent}>
                  <Text style={styles.tipTitle}>Medication reminders</Text>
                  <Text style={styles.tipDescription}>Try "When do I need to take my medicine?"</Text>
                </View>
              </View>
              <View style={styles.tipItem}>
                <Ionicons name="people" size={24} color="#4285F4" style={styles.tipIcon} />
                <View style={styles.tipContent}>
                  <Text style={styles.tipTitle}>Contact family</Text>
                  <Text style={styles.tipDescription}>Try "Call my daughter" or "Show me family photos"</Text>
                </View>
              </View>
              <View style={styles.tipItem}>
                <Ionicons name="help-buoy" size={24} color="#4285F4" style={styles.tipIcon} />
                <View style={styles.tipContent}>
                  <Text style={styles.tipTitle}>Get help</Text>
                  <Text style={styles.tipDescription}>Try "I need help" or "I'm feeling confused"</Text>
                </View>
              </View>
            </ScrollView>
          </Surface>
        )}
      </TouchableOpacity>
      
      {/* Cast Modal */}
      <Portal>
        <Dialog
          visible={castModalVisible}
          onDismiss={() => setCastModalVisible(false)}
          style={styles.castDialog}
        >
          <Dialog.Title>Cast to Device</Dialog.Title>
          <Dialog.Content>
            <Text style={styles.castDialogText}>
              Select a device to cast your AI companion:
            </Text>
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
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setCastModalVisible(false)}>Cancel</Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>
      
      {/* Mood Selection Modal */}
      <Portal>
        <Dialog
          visible={moodModalVisible}
          onDismiss={() => setMoodModalVisible(false)}
          style={styles.moodDialog}
        >
          <Dialog.Title>Change AI Personality</Dialog.Title>
          <Dialog.Content>
            <Text style={styles.moodDialogText}>
              Select a personality style for your AI companion:
            </Text>
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
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setMoodModalVisible(false)}>Cancel</Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F7FA',
  },
  fullScreenTouch: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatarContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatarWrapper: {
    width: width * 0.8,
    height: height * 0.6,
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatarImage: {
    width: '100%',
    height: '100%',
  },
  moodChip: {
    position: 'absolute',
    top: 20,
    backgroundColor: '#E8F1FF',
  },
  speakingIndicator: {
    position: 'absolute',
    bottom: 20,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'flex-end',
    height: 30,
    width: 60,
  },
  soundWave: {
    width: 4,
    backgroundColor: '#4285F4',
    marginHorizontal: 2,
    borderRadius: 2,
  },
  wave1: {
    height: 15,
    animationName: 'wave',
    animationDuration: '0.5s',
    animationIterationCount: 'infinite',
  },
  wave2: {
    height: 25,
    animationName: 'wave',
    animationDuration: '0.5s',
    animationDelay: '0.2s',
    animationIterationCount: 'infinite',
  },
  wave3: {
    height: 10,
    animationName: 'wave',
    animationDuration: '0.5s',
    animationDelay: '0.1s',
    animationIterationCount: 'infinite',
  },
  responseBubble: {
    position: 'absolute',
    top: 100,
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 20,
    maxWidth: width * 0.8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  responseText: {
    fontSize: 16,
    lineHeight: 22,
  },
  controlsOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'space-between',
    padding: 20,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
  },
  headerButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
  },
  headerActions: {
    flexDirection: 'row',
  },
  castingChip: {
    backgroundColor: '#E8F1FF',
    marginRight: 10,
  },
  microphoneContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  microphoneButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: '#4285F4',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 5,
  },
  listeningButton: {
    backgroundColor: '#FF3B30',
  },
  microphoneStatus: {
    fontSize: 16,
    color: '#333',
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  quickPhrasesContainer: {
    maxHeight: 60,
    marginBottom: 20,
  },
  quickPhrasesContent: {
    paddingHorizontal: 10,
    gap: 10,
  },
  quickPhraseButton: {
    backgroundColor: 'white',
    paddingHorizontal: 15,
    paddingVertical: 10,
    borderRadius: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  quickPhraseText: {
    fontSize: 14,
    color: '#4285F4',
  },
  bottomActions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
  },
  actionButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
  },
  tipsPanel: {
    position: 'absolute',
    bottom: 80,
    left: 20,
    right: 20,
    backgroundColor: 'white',
    borderRadius: 15,
    padding: 15,
    maxHeight: 300,
    elevation: 5,
  },
  tipsPanelHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  tipsPanelTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  tipsScrollView: {
    maxHeight: 220,
  },
  tipItem: {
    flexDirection: 'row',
    marginBottom: 15,
    alignItems: 'center',
  },
  tipIcon: {
    marginRight: 15,
  },
  tipContent: {
    flex: 1,
  },
  tipTitle: {
    fontSize: 16,
    fontWeight: '500',
    marginBottom: 4,
  },
  tipDescription: {
    fontSize: 14,
    color: '#777',
  },
  castDialog: {
    borderRadius: 15,
  },
  castDialogText: {
    marginBottom: 15,
  },
  castDevicesList: {
    gap: 10,
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
  moodDialog: {
    borderRadius: 15,
  },
  moodDialogText: {
    marginBottom: 15,
  },
  moodOptionsList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    gap: 10,
  },
  moodOption: {
    width: '48%',
    backgroundColor: '#F5F5F5',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 10,
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
}); 