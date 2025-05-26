import React, { useState, useRef, useEffect } from "react";
import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  Image,
  Animated,
  Modal,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  TextInput,
  Alert,
} from "react-native";
import { Ionicons, MaterialCommunityIcons } from "@expo/vector-icons";
import { Button, Surface, ActivityIndicator, Chip } from "react-native-paper";
import { SafeAreaView } from "react-native-safe-area-context";
import { reminderService } from "../services/reminderService";
import { userStatsService } from "../services/userStatsService";
import { aiService } from "../services/aiService";
import { voiceService, VoiceServiceCallbacks } from "../services/voiceService";

interface Message {
  id: string;
  text: string;
  sender: "user" | "ai";
  timestamp: Date;
}

interface TalkToAIScreenProps {
  setActiveTab: (tab: string) => void;
  setIsTalking?: (isTalking: boolean) => void;
  setAudioData?: (audioData: string) => void;
  isDesktopView?: boolean;
  voiceOnlyMode?: boolean;
}

export default function TalkToAIScreen({ 
  setActiveTab, 
  setIsTalking, 
  setAudioData,
  isDesktopView = false,
  voiceOnlyMode = false 
}: TalkToAIScreenProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [showControls, setShowControls] = useState(true);
  const scrollViewRef = useRef<ScrollView>(null);
  const lastNotificationTime = useRef<Record<string, number>>({});

  // Text-to-speech state (keeping this for AI responses)
  const [isSpeaking, setIsSpeaking] = useState(false);

  // Voice-to-text states
  const [isListening, setIsListening] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [isVoiceSupported, setIsVoiceSupported] = useState(false);
  const [availableProviders, setAvailableProviders] = useState<string[]>([]);
  const [currentProvider, setCurrentProvider] = useState<string>('');
  const pulseAnim = useRef(new Animated.Value(1)).current;

  // Voice recording animation
  useEffect(() => {
    if (isListening) {
      const pulse = () => {
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.2,
            duration: 600,
            useNativeDriver: Platform.OS !== 'web',
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 600,
            useNativeDriver: Platform.OS !== 'web',
          }),
        ]).start(() => {
          if (isListening) pulse();
        });
      };
      pulse();
    } else {
      pulseAnim.setValue(1);
    }
  }, [isListening, pulseAnim]);

  // Voice wave animation for listening state
  const waveAnim1 = useRef(new Animated.Value(0.3)).current;
  const waveAnim2 = useRef(new Animated.Value(0.5)).current;
  const waveAnim3 = useRef(new Animated.Value(0.8)).current;

  useEffect(() => {
    if (isListening) {
      const createWaveAnimation = (animValue: Animated.Value, delay: number) => {
        return Animated.loop(
          Animated.sequence([
            Animated.delay(delay),
            Animated.timing(animValue, {
              toValue: 1,
              duration: 400,
              useNativeDriver: Platform.OS !== 'web',
            }),
            Animated.timing(animValue, {
              toValue: 0.3,
              duration: 400,
              useNativeDriver: Platform.OS !== 'web',
            }),
          ])
        );
      };

      const wave1Animation = createWaveAnimation(waveAnim1, 0);
      const wave2Animation = createWaveAnimation(waveAnim2, 100);
      const wave3Animation = createWaveAnimation(waveAnim3, 200);

      wave1Animation.start();
      wave2Animation.start();
      wave3Animation.start();

      return () => {
        wave1Animation.stop();
        wave2Animation.stop();
        wave3Animation.stop();
      };
    } else {
      waveAnim1.setValue(0.3);
      waveAnim2.setValue(0.5);
      waveAnim3.setValue(0.8);
    }
  }, [isListening, waveAnim1, waveAnim2, waveAnim3]);

  // Initialize voice service
  useEffect(() => {
    const initializeVoiceService = () => {
      try {
        // Check if voice recognition is supported
        const isSupported = voiceService.isSpeechAvailable();
        setIsVoiceSupported(isSupported);

        if (!isSupported) {
          console.warn('No voice recognition providers available');
          return;
        }

        // Get available providers
        const providers = voiceService.getAvailableProviders();
        setAvailableProviders(providers);
        setCurrentProvider(providers[0] || ''); // Use the first (best) provider

        console.log('Available speech providers:', providers);

        // Set up voice service callbacks
        const callbacks: VoiceServiceCallbacks = {
          onSpeechStart: () => {
            console.log('Voice recognition started');
            setIsListening(true);
            setVoiceError(null);
          },
          onSpeechEnd: () => {
            console.log('Voice recognition ended');
            setIsListening(false);
          },
          onSpeechResult: (text: string) => {
            console.log('Voice recognition result:', text);
            setInputText(text);
            setIsListening(false);
            setVoiceError(null);
            
            // Automatically send the message if it's not empty
            if (text.trim()) {
              sendMessage(text.trim());
            }
          },
          onSpeechError: (error: string) => {
            console.error('Voice recognition error:', error);
            setIsListening(false);
            setVoiceError(error);
            
            // Show user-friendly error message based on error type
            if (error.includes('not-allowed') || error.includes('permission')) {
              Alert.alert(
                'Microphone Permission Required',
                'Please allow microphone access to use voice input. You may need to reload the page after granting permission.',
                [{ text: 'OK' }]
              );
            } else if (error.includes('no-speech')) {
              // Don't show alert for no speech detected, just show in UI
              setTimeout(() => setVoiceError(null), 3000);
            } else if (error.includes('network') || error.includes('connection') || error.includes('unavailable')) {
              // For network/connection errors, show a more helpful message
              setVoiceError('Speech service temporarily unavailable. Trying alternative provider...');
              
              // Try next available provider
              const nextProvider = providers.find(p => p !== currentProvider);
              if (nextProvider) {
                setCurrentProvider(nextProvider);
                console.log(`Switching to provider: ${nextProvider}`);
              }
              
              setTimeout(() => setVoiceError(null), 5000);
            } else {
              // For other errors, show a brief message
              setTimeout(() => setVoiceError(null), 5000);
            }
          },
        };

        voiceService.setCallbacks(callbacks);
      } catch (error) {
        console.error('Error initializing voice service:', error);
        setIsVoiceSupported(false);
      }
    };

    initializeVoiceService();

    // Cleanup on unmount
    return () => {
      try {
        voiceService.reset();
      } catch (error) {
        console.error('Error cleaning up voice service:', error);
      }
    };
  }, [currentProvider]);

  // Voice input handlers
  const startVoiceInput = async () => {
    try {
      setVoiceError(null);
      
      // Check if voice service is available before attempting
      if (!isVoiceSupported) {
        setVoiceError('Voice input not supported in this browser. Please use Chrome, Firefox, Safari, or Edge.');
        return;
      }
      
      // Use the current provider or let service choose the best one
      await voiceService.startListening(currentProvider);
    } catch (error) {
      console.error('Error starting voice input:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to start voice input';
      
      // Handle specific error types
      if (errorMessage.includes('network') || errorMessage.includes('connection')) {
        setVoiceError('Primary speech service unavailable. Trying backup service...');
        
        // Try with a different provider
        const fallbackProvider = availableProviders.find(p => p !== currentProvider);
        if (fallbackProvider) {
          setCurrentProvider(fallbackProvider);
          try {
            await voiceService.startListening(fallbackProvider);
            setVoiceError(null);
            return;
          } catch (fallbackError) {
            console.error('Fallback provider also failed:', fallbackError);
          }
        }
        
        setVoiceError('All speech services are currently unavailable. Please try again later.');
      } else if (errorMessage.includes('permission')) {
        setVoiceError('Microphone permission required. Please allow access and try again.');
      } else {
        setVoiceError(errorMessage);
      }
      
      // Clear error after delay
      setTimeout(() => setVoiceError(null), 5000);
    }
  };

  const stopVoiceInput = () => {
    try {
      voiceService.stopListening();
      setIsListening(false);
    } catch (error) {
      console.error('Error stopping voice input:', error);
      // Force reset the listening state even if stop fails
      setIsListening(false);
    }
  };

  // Sample quick phrases
  const quickPhrases = [
    "What day is it today?",
    "Remind me to take my medicine",
    "What's on my schedule?",
    "Call my caregiver",
    "Tell me about my family",
    "Help me remember where I put my glasses",
  ];

  // Keyboard shortcuts for voice input
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Space bar to start voice input (when not typing in input field)
      if (event.code === 'Space' && !event.repeat && isVoiceSupported && !isProcessing) {
        const activeElement = document.activeElement;
        const isInputFocused = activeElement?.tagName === 'INPUT' || activeElement?.tagName === 'TEXTAREA';
        
        if (!isInputFocused && !isListening) {
          event.preventDefault();
          startVoiceInput();
        }
      }
      
      // Escape to stop voice input
      if (event.code === 'Escape' && isListening) {
        event.preventDefault();
        stopVoiceInput();
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      // Release space bar to stop voice input (if it was used to start)
      if (event.code === 'Space' && isListening) {
        const activeElement = document.activeElement;
        const isInputFocused = activeElement?.tagName === 'INPUT' || activeElement?.tagName === 'TEXTAREA';
        
        if (!isInputFocused) {
          event.preventDefault();
          stopVoiceInput();
        }
      }
    };

    if (Platform.OS === 'web') {
      document.addEventListener('keydown', handleKeyDown);
      document.addEventListener('keyup', handleKeyUp);
      
      return () => {
        document.removeEventListener('keydown', handleKeyDown);
        document.removeEventListener('keyup', handleKeyUp);
      };
    }
  }, [isListening, isVoiceSupported, isProcessing]);

  // Check for upcoming reminders
  useEffect(() => {
    const checkUpcomingReminders = async () => {
      try {
        console.log("\n=== Checking Upcoming Reminders ===");

        // Get current time in Egyptian timezone
        const egyptianTimezone = "Africa/Cairo";
        const now = new Date();
        const egyptianTime = new Date(
          now.toLocaleString("en-US", { timeZone: egyptianTimezone })
        );
        console.log("Current time (Egypt):", egyptianTime.toLocaleString());

        // Get reminders using the correct endpoint
        console.log(
          "Fetching reminders from:",
          "https://www.zaaam.me/api/reminder"
        );
        const response = await fetch("https://www.zaaam.me/api/reminder", {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${localStorage.getItem("authToken")}`,
          },
        });

        console.log("Response status:", response.status);
        const data = await response.json();
        console.log("\n=== RAW API RESPONSE ===");
        console.log(JSON.stringify(data, null, 2));

        if (!data.success) {
          console.error("API returned error:", data.error);
          return;
        }

        if (!data.reminders || !Array.isArray(data.reminders)) {
          console.error("No reminders array in response:", data);
          return;
        }

        console.log("\n=== ALL REMINDERS DETAILS ===");
        console.log("Total reminders found:", data.reminders.length);

        data.reminders.forEach((reminder: any, index: number) => {
          // Convert reminder time to Egyptian timezone
          const reminderTime = new Date(reminder.start_time);
          const egyptianReminderTime = new Date(
            reminderTime.toLocaleString("en-US", { timeZone: egyptianTimezone })
          );

          // Calculate time difference
          const timeDiff =
            egyptianReminderTime.getTime() - egyptianTime.getTime();
          const hoursUntilReminder = Math.floor(timeDiff / (1000 * 60 * 60));
          const minutesUntilReminder = Math.floor(
            (timeDiff % (1000 * 60 * 60)) / (1000 * 60)
          );

          console.log(`\nReminder #${index + 1}:`);
          console.log("----------------------------------------");
          console.log("- Title:", reminder.title);
          console.log("- Original Time:", reminder.start_time);
          console.log(
            "- Egyptian Time:",
            egyptianReminderTime.toLocaleString()
          );
          console.log(
            "- Time Difference:",
            `${hoursUntilReminder} hours and ${minutesUntilReminder} minutes`
          );
          console.log(
            "- Description:",
            reminder.description || "No description"
          );
          console.log("- Status:", reminder.status || "active");
          console.log("- Created At:", reminder.created_at);

          if (timeDiff < 0) {
            console.log("- Status: PAST REMINDER");
          } else if (timeDiff <= 60 * 60 * 1000) {
            // Within 1 hour
            console.log("- Status: UPCOMING (within 1 hour)");
          } else {
            console.log("- Status: FUTURE REMINDER");
          }
          console.log("----------------------------------------");
        });

        const oneHourFromNow = new Date(
          egyptianTime.getTime() + 60 * 60 * 1000
        );

        console.log("\n=== TIME WINDOW CHECK ===");
        console.log("Current time:", egyptianTime.toLocaleString());
        console.log("Checking until:", oneHourFromNow.toLocaleString());

        let foundUpcomingReminders = false;

        // Check each reminder
        for (const reminder of data.reminders) {
          try {
            const reminderTime = new Date(reminder.start_time);
            const egyptianReminderTime = new Date(
              reminderTime.toLocaleString("en-US", {
                timeZone: egyptianTimezone,
              })
            );

            if (isNaN(egyptianReminderTime.getTime())) {
              console.error("Invalid reminder time:", reminder.start_time);
              continue;
            }

            const timeDiff =
              egyptianReminderTime.getTime() - egyptianTime.getTime();
            const minutesUntilReminder = Math.floor(timeDiff / (1000 * 60));

            // If reminder is within next hour and hasn't been notified in the last 5 minutes
            if (minutesUntilReminder >= 0 && minutesUntilReminder <= 60) {
              foundUpcomingReminders = true;
              console.log(`\nFound upcoming reminder: "${reminder.title}"`);
              console.log("- Time:", egyptianReminderTime.toLocaleString());
              console.log("- Minutes until reminder:", minutesUntilReminder);

              const lastNotified =
                lastNotificationTime.current[reminder._id] || 0;
              const fiveMinutesAgo = egyptianTime.getTime() - 5 * 60 * 1000;

              if (lastNotified < fiveMinutesAgo) {
                console.log("- Action: Sending notification");
                const notificationMessage: Message = {
                  id: Date.now().toString(),
                  text: `Hey, there is a meeting called "${reminder.title}" in the next hour!`,
                  sender: "ai",
                  timestamp: new Date(),
                };

                setMessages((prev) => [...prev, notificationMessage]);
                lastNotificationTime.current[reminder._id] =
                  egyptianTime.getTime();
              } else {
                console.log(
                  "- Action: Skipping notification (already notified in last 5 minutes)"
                );
              }
            }
          } catch (reminderError) {
            console.error(
              "Error processing reminder:",
              reminder,
              reminderError
            );
          }
        }

        if (!foundUpcomingReminders) {
          console.log("\nNo upcoming reminders found within the next hour");
        }

        console.log("\n=== End of Reminder Check ===\n");
      } catch (error) {
        console.error("Error checking upcoming reminders:", error);
      }
    };

    // Check immediately on mount
    console.log("Starting reminder check interval");
    checkUpcomingReminders();

    // Function to get random interval between 15-30 minutes
    const getRandomInterval = () => {
      const minMinutes = 15;
      const maxMinutes = 30;
      const randomMinutes =
        Math.floor(Math.random() * (maxMinutes - minMinutes + 1)) + minMinutes;
      console.log(`Next check will be in ${randomMinutes} minutes`);
      return randomMinutes * 60 * 1000; // Convert to milliseconds
    };

    // Set up interval with random timing
    let intervalId: NodeJS.Timeout;
    const scheduleNextCheck = () => {
      intervalId = setTimeout(() => {
        checkUpcomingReminders();
        scheduleNextCheck(); // Schedule the next check
      }, getRandomInterval());
    };

    // Start the first random interval
    scheduleNextCheck();

    // Cleanup interval on unmount
    return () => {
      console.log("Cleaning up reminder check interval");
      clearTimeout(intervalId);
    };
  }, []);

  const speakResponse = async (text: string) => {
    try {
      setIsSpeaking(true);
      if (setIsTalking) {
        setIsTalking(true);
      }
      
      await voiceService.speak(text, {
        language: 'en-US',
        pitch: 1.0,
        rate: 0.8,
      });
    } catch (error) {
      console.error('Error speaking response:', error);
    } finally {
      setIsSpeaking(false);
      if (setIsTalking) {
        setIsTalking(false);
      }
    }
  };

  const stopSpeaking = async () => {
    try {
      await voiceService.stopSpeaking();
      setIsSpeaking(false);
      if (setIsTalking) {
        setIsTalking(false);
      }
    } catch (error) {
      console.error('Error stopping speech:', error);
    }
  };

  const sendMessage = async (text: string) => {
    // Stop any active voice input
    if (isListening) {
      stopVoiceInput();
    }
    
    // Activate avatar talking if prop is provided
    if (setIsTalking) {
      setIsTalking(true);
    }
    
    if (!text.trim()) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text: text.trim(),
      sender: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputText("");
    setIsProcessing(true);
    setVoiceError(null); // Clear any voice errors

    try {
      // Use the main AI service which now includes audio support
      const aiResponse = await aiService.processAIRequest(text.trim());

      const responseText = aiResponse.response || "I'm here to help you. What would you like to know?";
      
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: responseText,
        sender: "ai",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, aiMessage]);
      
      // Set the audio data for the avatar if audio is present
      if (setAudioData && aiResponse.audio) {
        setAudioData(aiResponse.audio);
      }

      // Automatically speak the AI response
      await speakResponse(responseText);

      // Set a timeout to stop the avatar talking animation after a delay
      // proportional to the response length
      if (setIsTalking) {
        const talkingDuration = Math.min(Math.max(responseText.length * 50, 2000), 10000);
        setTimeout(() => {
          setIsTalking(false);
        }, talkingDuration);
      }

      // Increment AI interactions
      userStatsService.incrementAIInteractions();
    } catch (error) {
      console.error("Error processing message:", error);
      // Add error message
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "I'm sorry, I couldn't process your request right now. Please try again.",
        sender: "ai",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
      
      // Stop talking animation on error
      if (setIsTalking) {
        setIsTalking(false);
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSend = () => {
    if (inputText.trim()) {
      sendMessage(inputText);
    }
  };

  useEffect(() => {
    if (scrollViewRef.current) {
      scrollViewRef.current.scrollToEnd({ animated: true });
    }
  }, [messages]);

  return (
    <SafeAreaView style={styles.container}>
      {!voiceOnlyMode && (
        <>
          <ScrollView
            ref={scrollViewRef}
            style={styles.messagesContainer}
            contentContainerStyle={styles.messagesContent}
            onContentSizeChange={() => scrollViewRef.current?.scrollToEnd({ animated: true })}
          >
            {messages.map((message) => (
              <Surface
                key={message.id}
                style={[
                  styles.messageBubble,
                  message.sender === "user"
                    ? styles.userMessage
                    : styles.aiMessage,
                ]}
              >
                <Text style={styles.messageText}>{message.text}</Text>
              </Surface>
            ))}
          </ScrollView>

          <KeyboardAvoidingView
            behavior={Platform.OS === "ios" ? "padding" : "height"}
            style={styles.inputContainer}
          >
            <TextInput
              style={styles.input}
              value={inputText}
              onChangeText={setInputText}
              placeholder="Type your message..."
              multiline
            />
            <TouchableOpacity
              style={styles.sendButton}
              onPress={handleSend}
              disabled={isProcessing || !inputText.trim()}
            >
              <Ionicons
                name="send"
                size={24}
                color={isProcessing || !inputText.trim() ? "#666" : "#007AFF"}
              />
            </TouchableOpacity>
          </KeyboardAvoidingView>
        </>
      )}

      {/* Voice input button - always shown in desktop mode or when voiceOnlyMode is true */}
      <TouchableOpacity
        style={[
          styles.voiceButton,
          isListening && styles.voiceButtonActive,
          voiceOnlyMode && styles.voiceButtonLarge
        ]}
        onPress={isListening ? stopVoiceInput : startVoiceInput}
      >
        <MaterialCommunityIcons
          name={isListening ? "microphone" : "microphone-outline"}
          size={voiceOnlyMode ? 36 : 24}
          color={isListening ? "#FF3B30" : "#007AFF"}
        />
      </TouchableOpacity>

      {voiceError && (
        <Text style={styles.errorText}>{voiceError}</Text>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  keyboardAvoidingView: {
    flex: 1,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: 16,
    backgroundColor: "#fff",
    borderBottomWidth: 1,
    borderBottomColor: "#eee",
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: "bold",
  },
  headerRight: {
    flexDirection: "row",
    alignItems: "center",
  },
  messagesContainer: {
    flex: 1,
    padding: 10,
  },
  messagesContent: {
    flexGrow: 1,
  },
  messageBubble: {
    maxWidth: '80%',
    padding: 10,
    marginVertical: 5,
    borderRadius: 15,
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#007AFF',
  },
  aiMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#E9E9EB',
  },
  messageText: {
    fontSize: 16,
    color: '#000000',
  },
  userMessageText: {
    color: "#fff",
  },
  aiMessageText: {
    color: "#000",
  },
  timestamp: {
    fontSize: 12,
    color: "#666",
    marginTop: 4,
    alignSelf: "flex-end",
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 10,
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: '#E9E9EB',
  },
  input: {
    flex: 1,
    marginRight: 10,
    padding: 10,
    backgroundColor: '#F2F2F7',
    borderRadius: 20,
    maxHeight: 100,
  },
  sendButton: {
    padding: 10,
  },
  quickPhrasesContainer: {
    padding: 8,
    backgroundColor: "#fff",
  },
  quickPhrase: {
    backgroundColor: "#f0f0f0",
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 16,
    marginRight: 8,
  },
  quickPhraseText: {
    fontSize: 14,
    color: "#333",
  },
  sendButtonDisabled: {
    backgroundColor: "#ccc",
  },
  speakingIndicator: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 4,
  },
  speakingText: {
    fontSize: 12,
    color: "#007AFF",
    marginLeft: 4,
  },
  instructionsContainer: {
    padding: 16,
    backgroundColor: "#fff",
    borderTopWidth: 1,
    borderTopColor: "#eee",
  },
  instructionsText: {
    fontSize: 14,
    color: "#333",
  },
  voiceInstructionsText: {
    fontSize: 12,
    color: "#007AFF",
    marginTop: 4,
    fontStyle: "italic",
  },
  keyboardShortcutText: {
    fontSize: 12,
    color: "#007AFF",
    marginTop: 4,
    fontStyle: "italic",
  },
  voiceUnsupportedText: {
    fontSize: 12,
    color: "#ff4444",
    marginTop: 4,
    fontStyle: "italic",
  },
  errorContainer: {
    flexDirection: "row",
    alignItems: "center",
    padding: 12,
    backgroundColor: "#fff5f5",
    borderWidth: 1,
    borderColor: "#ffcccc",
    borderRadius: 8,
    marginHorizontal: 16,
    marginBottom: 8,
  },
  errorText: {
    color: '#FF3B30',
    textAlign: 'center',
    marginTop: 10,
  },
  voiceButton: {
    position: 'absolute',
    bottom: 20,
    right: 20,
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#FFFFFF',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  voiceButtonActive: {
    backgroundColor: '#FFE5E5',
  },
  stopSpeakingButton: {
    backgroundColor: "#ff4444",
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 8,
  },
  voiceStatus: {
    flexDirection: "row",
    alignItems: "center",
    marginRight: 8,
    backgroundColor: "#fff5f5",
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#ffcccc",
  },
  voiceStatusText: {
    fontSize: 11,
    color: "#ff4444",
    marginLeft: 4,
    fontWeight: "600",
  },
  voiceWaveContainer: {
    position: "absolute",
    top: "50%",
    left: 20,
    right: 80,
    height: 40,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-evenly",
    zIndex: 1,
  },
  voiceWave: {
    width: 3,
    height: 20,
    backgroundColor: "#007AFF",
    borderRadius: 2,
    marginHorizontal: 2,
  },
  voiceUnavailableText: {
    fontSize: 12,
    color: "#ff4444",
    marginTop: 4,
    fontStyle: "italic",
  },
  providerInfoText: {
    fontSize: 12,
    color: "#007AFF",
    marginTop: 4,
    fontStyle: "italic",
  },
  voiceButtonLarge: {
    width: 80,
    height: 80,
    borderRadius: 40,
  },
});
