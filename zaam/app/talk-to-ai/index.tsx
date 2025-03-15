import React, { useState, useRef } from 'react';
import { StyleSheet, View, ScrollView, KeyboardAvoidingView, Platform } from 'react-native';
import { Text, IconButton, TextInput, Surface, Avatar, Button } from 'react-native-paper';
import { router } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { WebView } from 'react-native-webview';

type Message = {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: Date;
};

export default function TalkToAIScreen() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! How can I assist you today?',
      sender: 'ai',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isCasting, setIsCasting] = useState(false);
  const scrollViewRef = useRef<ScrollView>(null);

  const handleSend = () => {
    if (!input.trim()) return;

    const newMessage: Message = {
      id: Date.now().toString(),
      text: input,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages([...messages, newMessage]);
    setInput('');

    // Simulate AI response
    setTimeout(() => {
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: 'I understand. I\'m here to help you with that.',
        sender: 'ai',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, aiResponse]);
    }, 1000);
  };

  const toggleRecording = () => {
    setIsRecording(!isRecording);
    // TODO: Implement actual voice recording
  };

  const toggleCasting = () => {
    setIsCasting(!isCasting);
    // TODO: Implement actual TV casting
  };

  // Simple 3D model viewer using Three.js
  const threejsContent = `
    <html>
      <head>
        <style>
          body { margin: 0; }
          canvas { width: 100%; height: 100%; }
        </style>
      </head>
      <body>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
          const scene = new THREE.Scene();
          const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
          const renderer = new THREE.WebGLRenderer();
          renderer.setSize(window.innerWidth, window.innerHeight);
          document.body.appendChild(renderer.domElement);

          const geometry = new THREE.SphereGeometry(1, 32, 32);
          const material = new THREE.MeshPhongMaterial({ color: 0x00ff00 });
          const sphere = new THREE.Mesh(geometry, material);
          scene.add(sphere);

          const light = new THREE.PointLight(0xffffff, 1, 100);
          light.position.set(10, 10, 10);
          scene.add(light);

          camera.position.z = 5;

          function animate() {
            requestAnimationFrame(animate);
            sphere.rotation.x += 0.01;
            sphere.rotation.y += 0.01;
            renderer.render(scene, camera);
          }
          animate();
        </script>
      </body>
    </html>
  `;

  return (
    <View style={styles.container}>
      <StatusBar style="light" />

      {/* Header */}
      <Surface style={styles.header} elevation={2}>
        <View style={styles.headerContent}>
          <IconButton icon="arrow-left" size={24} onPress={() => router.back()} />
          <Text variant="titleLarge" style={styles.title}>AI Assistant</Text>
          <IconButton
            icon={isCasting ? 'cast-connected' : 'cast'}
            size={24}
            onPress={toggleCasting}
          />
        </View>
      </Surface>

      {/* 3D AI Model */}
      <View style={styles.modelContainer}>
        <WebView
          source={{ html: threejsContent }}
          style={styles.webview}
          javaScriptEnabled={true}
        />
      </View>

      {/* Chat Messages */}
      <KeyboardAvoidingView 
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.chatContainer}
        keyboardVerticalOffset={90}
      >
        <ScrollView
          ref={scrollViewRef}
          style={styles.messagesContainer}
          contentContainerStyle={styles.messagesContent}
          onContentSizeChange={() => scrollViewRef.current?.scrollToEnd({ animated: true })}
        >
          {messages.map((message) => (
            <View
              key={message.id}
              style={[
                styles.messageBubble,
                message.sender === 'user' ? styles.userMessage : styles.aiMessage,
              ]}
            >
              {message.sender === 'ai' && (
                <Avatar.Icon size={24} icon="robot" style={styles.avatar} />
              )}
              <View style={styles.messageContent}>
                <Text style={styles.messageText}>{message.text}</Text>
                <Text style={styles.timestamp}>
                  {message.timestamp.toLocaleTimeString([], { 
                    hour: '2-digit', 
                    minute: '2-digit' 
                  })}
                </Text>
              </View>
            </View>
          ))}
        </ScrollView>

        {/* Input Area */}
        <Surface style={styles.inputContainer} elevation={2}>
          <TextInput
            value={input}
            onChangeText={setInput}
            placeholder="Type a message..."
            mode="outlined"
            style={styles.input}
            right={
              <TextInput.Icon
                icon={isRecording ? 'microphone' : 'microphone-outline'}
                onPress={toggleRecording}
                forceTextInputFocus={false}
              />
            }
          />
          <Button
            mode="contained"
            onPress={handleSend}
            style={styles.sendButton}
            disabled={!input.trim()}
          >
            Send
          </Button>
        </Surface>
      </KeyboardAvoidingView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: '#007AFF',
    paddingTop: 60,
    paddingBottom: 20,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
  },
  title: {
    flex: 1,
    color: '#fff',
    marginLeft: 16,
    fontWeight: 'bold',
  },
  modelContainer: {
    height: 300,
    backgroundColor: '#000',
  },
  webview: {
    flex: 1,
  },
  chatContainer: {
    flex: 1,
  },
  messagesContainer: {
    flex: 1,
  },
  messagesContent: {
    padding: 16,
  },
  messageBubble: {
    flexDirection: 'row',
    marginBottom: 12,
    maxWidth: '80%',
  },
  userMessage: {
    alignSelf: 'flex-end',
    flexDirection: 'row-reverse',
  },
  aiMessage: {
    alignSelf: 'flex-start',
  },
  avatar: {
    backgroundColor: '#007AFF',
    marginRight: 8,
    marginTop: 4,
  },
  messageContent: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 12,
    elevation: 1,
  },
  messageText: {
    fontSize: 16,
  },
  timestamp: {
    fontSize: 12,
    opacity: 0.5,
    marginTop: 4,
    alignSelf: 'flex-end',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#fff',
  },
  input: {
    flex: 1,
    marginRight: 8,
    backgroundColor: '#fff',
  },
  sendButton: {
    borderRadius: 20,
  },
}); 