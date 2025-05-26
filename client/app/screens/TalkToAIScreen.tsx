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
}

export default function TalkToAIScreen({ setActiveTab, setIsTalking, setAudioData }: TalkToAIScreenProps) {
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
          console.warn('Voice recognition not supported in this environment');
          return;
        }

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
              setVoiceError('Speech service temporarily unavailable. Please try again.');
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
  }, []);

  // Voice input handlers
  const startVoiceInput = async () => {
    try {
      setVoiceError(null);
      
      // Check if voice service is available before attempting
      if (!isVoiceSupported) {
        setVoiceError('Voice input not supported in this browser. Please use Chrome, Firefox, Safari, or Edge.');
        return;
      }
      
      await voiceService.startListening();
    } catch (error) {
      console.error('Error starting voice input:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to start voice input';
      
      // Handle specific error types
      if (errorMessage.includes('network') || errorMessage.includes('connection')) {
        setVoiceError('Speech service is currently unavailable. Please try again in a moment.');
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
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={styles.keyboardAvoidingView}
      >
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Talk to AI</Text>
          <View style={styles.headerRight}>
            {/* Voice Status Indicator */}
            {isListening && (
              <View style={styles.voiceStatus}>
                <Ionicons name="mic" size={16} color="#ff4444" />
                <Text style={styles.voiceStatusText}>Listening...</Text>
              </View>
            )}
            <TouchableOpacity onPress={() => setShowControls(!showControls)}>
              <Ionicons
                name={showControls ? "chevron-up" : "chevron-down"}
                size={24}
                color="#000"
              />
            </TouchableOpacity>
          </View>
        </View>

        {/* Instructions */}
        {showControls && (
          <View style={styles.instructionsContainer}>
            <Text style={styles.instructionsText}>
              💬 Type your message below • 🎙️ Use voice input • 🔊 AI responses will be spoken automatically
            </Text>
            {isVoiceSupported && (
              <View>
                <Text style={styles.voiceInstructionsText}>
                  Tap the microphone to speak your message
                </Text>
                <Text style={styles.keyboardShortcutText}>
                  💡 Pro tip: Hold spacebar to use voice input, Escape to stop
                </Text>
                {voiceError && voiceError.includes('unavailable') && (
                  <Text style={styles.voiceUnavailableText}>
                    ⚠️ Voice service temporarily unavailable - you can still type your message
                  </Text>
                )}
              </View>
            )}
            {!isVoiceSupported && (
              <Text style={styles.voiceUnsupportedText}>
                Voice input requires HTTPS and a modern browser (Chrome, Firefox, Safari, Edge)
              </Text>
            )}
          </View>
        )}

        {/* Voice Error Display */}
        {voiceError && (
          <View style={styles.errorContainer}>
            <Ionicons name="warning-outline" size={20} color="#ff4444" />
            <Text style={styles.errorText}>{voiceError}</Text>
          </View>
        )}

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
                message.sender === "user" ? styles.userBubble : styles.aiBubble,
              ]}
            >
              <View style={styles.messageContent}>
                <Text style={[
                  styles.messageText,
                  message.sender === "user" ? styles.userMessageText : styles.aiMessageText
                ]}>
                  {message.text}
                </Text>
                {message.sender === "ai" && isSpeaking && (
                  <View style={styles.speakingIndicator}>
                    <Ionicons name="volume-high" size={16} color="#007AFF" />
                    <Text style={styles.speakingText}>Speaking...</Text>
                  </View>
                )}
              </View>
              <Text style={styles.timestamp}>
                {message.timestamp.toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </Text>
            </View>
          ))}
          {isProcessing && (
            <View style={styles.messageBubble}>
              <ActivityIndicator size="small" color="#666" />
            </View>
          )}
        </ScrollView>

        <View style={styles.inputContainer}>
          {/* Voice Wave Animation */}
          {isListening && (
            <View style={styles.voiceWaveContainer}>
              <Animated.View
                style={[
                  styles.voiceWave,
                  { transform: [{ scaleY: waveAnim1 }] }
                ]}
              />
              <Animated.View
                style={[
                  styles.voiceWave,
                  { transform: [{ scaleY: waveAnim2 }] }
                ]}
              />
              <Animated.View
                style={[
                  styles.voiceWave,
                  { transform: [{ scaleY: waveAnim3 }] }
                ]}
              />
              <Animated.View
                style={[
                  styles.voiceWave,
                  { transform: [{ scaleY: waveAnim2 }] }
                ]}
              />
              <Animated.View
                style={[
                  styles.voiceWave,
                  { transform: [{ scaleY: waveAnim1 }] }
                ]}
              />
            </View>
          )}
          
          <TextInput
            style={styles.input}
            value={inputText}
            onChangeText={setInputText}
            placeholder={isListening ? "Listening..." : "Type your message or use voice input..."}
            placeholderTextColor="#666"
            multiline
            editable={!isListening}
          />
          
          {/* Voice Input Button */}
          {isVoiceSupported && (
            <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
              <TouchableOpacity
                style={[
                  styles.voiceButton,
                  isListening && styles.voiceButtonActive
                ]}
                onPress={isListening ? stopVoiceInput : startVoiceInput}
                disabled={isProcessing}
              >
                <Ionicons 
                  name={isListening ? "stop" : "mic"} 
                  size={24} 
                  color={isListening ? "#fff" : "#007AFF"} 
                />
              </TouchableOpacity>
            </Animated.View>
          )}
          
          {/* Stop Speaking Button */}
          {isSpeaking && (
            <TouchableOpacity
              style={styles.stopSpeakingButton}
              onPress={stopSpeaking}
            >
              <Ionicons name="volume-mute" size={24} color="#ff4444" />
            </TouchableOpacity>
          )}
          
          <TouchableOpacity
            style={[
              styles.sendButton,
              (!inputText.trim() || isProcessing || isListening) && styles.sendButtonDisabled
            ]}
            onPress={handleSend}
            disabled={!inputText.trim() || isProcessing || isListening}
          >
            <Ionicons name="send" size={24} color="#fff" />
          </TouchableOpacity>
        </View>

        {showControls && (
          <View style={styles.quickPhrasesContainer}>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              {quickPhrases.map((phrase, index) => (
                <TouchableOpacity
                  key={index}
                  style={styles.quickPhrase}
                  onPress={() => sendMessage(phrase)}
                >
                  <Text style={styles.quickPhraseText}>{phrase}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
        )}
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f5f5f5",
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
  },
  messagesContent: {
    padding: 16,
  },
  messageBubble: {
    maxWidth: "80%",
    padding: 12,
    borderRadius: 16,
    marginBottom: 8,
  },
  userBubble: {
    backgroundColor: "#007AFF",
    alignSelf: "flex-end",
  },
  aiBubble: {
    backgroundColor: "#fff",
    alignSelf: "flex-start",
    borderWidth: 1,
    borderColor: "#eee",
  },
  messageContent: {
    flexDirection: "column",
    flex: 1,
  },
  messageText: {
    fontSize: 16,
    color: "#000",
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
    flexDirection: "row",
    padding: 16,
    backgroundColor: "#fff",
    borderTopWidth: 1,
    borderTopColor: "#eee",
  },
  input: {
    flex: 1,
    backgroundColor: "#f0f0f0",
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginRight: 8,
    maxHeight: 100,
  },
  sendButton: {
    backgroundColor: "#007AFF",
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: "center",
    alignItems: "center",
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
    fontSize: 13,
    color: "#cc0000",
    marginLeft: 8,
    flex: 1,
  },
  voiceButton: {
    backgroundColor: "#f0f0f0",
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 8,
    borderWidth: 2,
    borderColor: "#007AFF",
  },
  voiceButtonActive: {
    backgroundColor: "#ff4444",
    borderColor: "#ff4444",
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
});
