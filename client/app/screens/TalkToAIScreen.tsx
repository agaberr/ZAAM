import React, { useState, useRef } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, Image, Animated, Modal, ScrollView } from 'react-native';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { Button, Surface, ActivityIndicator, Chip } from 'react-native-paper';

export default function TalkToAIScreen({ setActiveTab }) {
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

  const processUserInput = (input) => {
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
      setIsSpeaking(true);
      
      // Simulate speaking
      setTimeout(() => {
        setIsSpeaking(false);
      }, 3000);
    }, 1500);
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
    <View style={styles.container}>
      <TouchableOpacity 
        activeOpacity={1} 
        style={styles.fullScreenTouch}
        onPress={() => setShowControls(true)}
      >
        {/* 3D AI Avatar */}
        <View style={styles.avatarContainer}>
          <Animated.View 
            style={[
              styles.avatarWrapper,
              { transform: [{ scale: pulseAnim }] }
            ]}
          >
            <Image 
              source={require('../assets/ai-avatar.png')} 
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
        
        {/* Conversation bubble for latest AI response */}
        {aiResponse && (
          <View style={styles.responseBubble}>
            <Text style={styles.responseText}>{aiResponse}</Text>
          </View>
        )}
        
        {/* Controls overlay */}
        {showControls && (
          <View style={styles.controlsOverlay}>
            {/* Header with options */}
            <View style={styles.header}>
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
                  <TouchableOpacity
                    style={styles.castButton}
                    onPress={handleCastPress}
                  >
                    <Ionicons name="tv-outline" size={24} color="#4285F4" />
                    <Text style={styles.castText}>Cast to TV</Text>
                  </TouchableOpacity>
                )}
                <TouchableOpacity
                  onPress={() => setMoodModalVisible(true)}
                >
                  <Ionicons name="options-outline" size={24} color="#4285F4" />
                </TouchableOpacity>
              </View>
            </View>
            
            {/* Microphone button and status */}
            <View style={styles.microphoneContainer}>
              <TouchableOpacity 
                style={[
                  styles.micButton,
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
              <Text style={styles.helpText}>
                {isListening ? 'Listening...' : 'Tap to speak with your AI companion'}
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

            {/* Tips button */}
            <TouchableOpacity 
              style={styles.tipsButton}
              onPress={() => setShowTips(!showTips)}
            >
              <Ionicons name="information-circle-outline" size={24} color="#4285F4" />
              <Text style={styles.tipsText}>
                {showTips ? 'Hide Tips' : 'Show Tips'}
              </Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Tips panel */}
        {showTips && (
          <Surface style={styles.tipsPanel}>
            <View style={styles.tipsPanelHeader}>
              <Text style={styles.tipsPanelTitle}>Tips for talking to your AI</Text>
              <TouchableOpacity onPress={() => setShowTips(false)}>
                <Ionicons name="close" size={24} color="#4285F4" />
              </TouchableOpacity>
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
            </ScrollView>
          </Surface>
        )}

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
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5F7FA',
  },
  fullScreenTouch: {
    flex: 1,
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatarContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatarWrapper: {
    position: 'relative',
  },
  avatarImage: {
    width: 300,
    height: 300,
  },
  speakingIndicator: {
    position: 'absolute',
    bottom: 20,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'center',
  },
  soundWave: {
    width: 8,
    height: 20,
    borderRadius: 4,
    marginHorizontal: 2,
  },
  wave1: {
    height: 15,
    backgroundColor: '#4285F4',
  },
  wave2: {
    height: 25,
    backgroundColor: '#FF9500',
  },
  wave3: {
    height: 10,
    backgroundColor: '#FF3B30',
  },
  moodChip: {
    backgroundColor: '#E8F1FF',
    marginTop: 10,
  },
  responseBubble: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 20,
    maxWidth: '80%',
    marginBottom: 20,
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
    justifyContent: 'flex-end',
    width: '100%',
    paddingTop: 20,
  },
  headerActions: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  castButton: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 15,
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
  },
  castText: {
    marginLeft: 8,
    color: '#4285F4',
    fontWeight: '500',
  },
  castingChip: {
    backgroundColor: '#E8F1FF',
    marginRight: 15,
  },
  microphoneContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  micButton: {
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
  helpText: {
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
  tipsButton: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
    marginBottom: 20,
  },
  tipsText: {
    marginLeft: 8,
    color: '#4285F4',
    fontWeight: '500',
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
}); 