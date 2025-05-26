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

  // Voice-related state
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speechAvailable, setSpeechAvailable] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);

  // Sample quick phrases
  const quickPhrases = [
    "What day is it today?",
    "Remind me to take my medicine",
    "What's on my schedule?",
    "Call my caregiver",
    "Tell me about my family",
    "Help me remember where I put my glasses",
  ];

  // Initialize voice service
  useEffect(() => {
    const initializeVoice = () => {
      setSpeechAvailable(voiceService.isSpeechAvailable());
      
      const callbacks: VoiceServiceCallbacks = {
        onSpeechStart: () => {
          setIsListening(true);
          setVoiceError(null);
        },
        onSpeechEnd: () => {
          setIsListening(false);
        },
        onSpeechResult: (text: string) => {
          setInputText(text);
          setIsListening(false);
          // Automatically send the message
          sendMessage(text);
        },
        onSpeechError: (error: string) => {
          setIsListening(false);
          setVoiceError(error);
          console.error('Speech recognition error:', error);
        },
      };
      
      voiceService.setCallbacks(callbacks);
    };

    initializeVoice();
  }, []);

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

  // Voice control functions
  const startVoiceRecording = async () => {
    try {
      if (!speechAvailable) {
        Alert.alert('Speech Recognition', 'Speech recognition is not available on this device.');
        return;
      }
      
      await voiceService.startListening();
    } catch (error) {
      console.error('Error starting voice recording:', error);
      Alert.alert('Error', 'Failed to start voice recording. Please try again.');
    }
  };

  const stopVoiceRecording = () => {
    voiceService.stopListening();
  };

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
            {speechAvailable && (
              <View style={styles.voiceStatus}>
                <Ionicons name="mic" size={16} color="#4CAF50" />
                <Text style={styles.voiceStatusText}>Voice Ready</Text>
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

        {/* Voice Instructions */}
        {speechAvailable && showControls && (
          <View style={styles.instructionsContainer}>
            <Text style={styles.instructionsText}>
              🎤 Tap the microphone to speak • 🔊 AI responses will be spoken automatically
            </Text>
          </View>
        )}

        {/* Voice Troubleshooting */}
        {!speechAvailable && showControls && (
          <View style={styles.troubleshootingContainer}>
            <Text style={styles.troubleshootingTitle}>🔧 Voice Setup Required</Text>
            <Text style={styles.troubleshootingText}>
              To enable voice features:
            </Text>
            <Text style={styles.troubleshootingText}>
              • Use HTTPS: Open https://localhost:8081 instead
            </Text>
            <Text style={styles.troubleshootingText}>
              • Allow microphone permissions when prompted
            </Text>
            <Text style={styles.troubleshootingText}>
              • Ensure you have internet connection
            </Text>
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
          <TextInput
            style={styles.input}
            value={inputText}
            onChangeText={setInputText}
            placeholder={isListening ? "Listening..." : "Type your message..."}
            placeholderTextColor="#666"
            multiline
            editable={!isListening}
          />
          
          {/* Voice Recording Button */}
          {speechAvailable && (
            <TouchableOpacity
              style={[
                styles.voiceButton,
                isListening && styles.voiceButtonActive
              ]}
              onPress={isListening ? stopVoiceRecording : startVoiceRecording}
              disabled={isProcessing}
            >
              <Ionicons
                name={isListening ? "stop" : "mic"}
                size={24}
                color={isListening ? "#ff4444" : "#007AFF"}
              />
            </TouchableOpacity>
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
              (!inputText.trim() || isListening || isProcessing) && styles.sendButtonDisabled
            ]}
            onPress={handleSend}
            disabled={!inputText.trim() || isListening || isProcessing}
          >
            <Ionicons name="send" size={24} color="#fff" />
          </TouchableOpacity>
        </View>

        {/* Voice Error Display */}
        {voiceError && (
          <View style={styles.errorContainer}>
            <Text style={styles.errorText}>Voice Error: {voiceError}</Text>
            <TouchableOpacity onPress={() => setVoiceError(null)}>
              <Ionicons name="close" size={20} color="#ff4444" />
            </TouchableOpacity>
          </View>
        )}

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
  voiceStatus: {
    flexDirection: "row",
    alignItems: "center",
    marginRight: 8,
  },
  voiceStatusText: {
    fontSize: 14,
    color: "#333",
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
  voiceButton: {
    backgroundColor: "#007AFF",
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 8,
  },
  voiceButtonActive: {
    backgroundColor: "#ff4444",
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
  sendButtonDisabled: {
    backgroundColor: "#ccc",
  },
  errorContainer: {
    flexDirection: "row",
    alignItems: "center",
    padding: 16,
    backgroundColor: "#fff",
    borderTopWidth: 1,
    borderTopColor: "#eee",
  },
  errorText: {
    flex: 1,
    color: "#ff4444",
    fontSize: 14,
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
  troubleshootingContainer: {
    padding: 16,
    backgroundColor: "#fff3cd",
    borderTopWidth: 1,
    borderTopColor: "#eee",
    borderLeftWidth: 4,
    borderLeftColor: "#ffc107",
  },
  troubleshootingTitle: {
    fontSize: 16,
    fontWeight: "bold",
    marginBottom: 8,
    color: "#856404",
  },
  troubleshootingText: {
    fontSize: 14,
    color: "#856404",
    marginBottom: 4,
  },
});
