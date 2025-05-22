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
} from "react-native";
import { Ionicons, MaterialCommunityIcons } from "@expo/vector-icons";
import { Button, Surface, ActivityIndicator, Chip } from "react-native-paper";
import { SafeAreaView } from "react-native-safe-area-context";
import { reminderService } from "../services/reminderService";

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

  // Sample quick phrases
  const quickPhrases = [
    "What day is it today?",
    "Remind me to take my medicine",
    "What's on my schedule?",
    "Call my caregiver",
    "Tell me about my family",
    "Help me remember where I put my glasses",
  ];

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
          "http://localhost:5000/api/reminder"
        );
        const response = await fetch("http://localhost:5000/api/reminder", {
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
      // Call the AI processing endpoint
      const response = await fetch("http://localhost:5000/api/ai/process", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: text.trim() }),
      });

      const data = await response.json();

      const responseText = data.response || "I'm here to help you. What would you like to know?";
      
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: responseText,
        sender: "ai",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, aiMessage]);
      
      // Provide dummy audio data for the avatar if prop is provided
      if (setAudioData) {
        setAudioData("dummyAudioData");
      }

      // Set a timeout to stop the avatar talking animation after a delay
      // proportional to the response length
      if (setIsTalking) {
        const talkingDuration = Math.min(Math.max(responseText.length * 50, 2000), 10000);
        setTimeout(() => {
          setIsTalking(false);
        }, talkingDuration);
      }
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
          <TouchableOpacity onPress={() => setShowControls(!showControls)}>
            <Ionicons
              name={showControls ? "chevron-up" : "chevron-down"}
              size={24}
              color="#000"
            />
          </TouchableOpacity>
        </View>

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
              <Text style={styles.messageText}>{message.text}</Text>
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
            placeholder="Type your message..."
            placeholderTextColor="#666"
            multiline
          />
          <TouchableOpacity style={styles.sendButton} onPress={handleSend}>
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
  messageText: {
    fontSize: 16,
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
});
