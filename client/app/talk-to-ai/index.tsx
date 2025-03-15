import React, { useState, useRef } from 'react';
import { StyleSheet, View, ScrollView, KeyboardAvoidingView, Platform } from 'react-native';
import { Text, Card, IconButton, TextInput, Avatar, Surface, useTheme } from 'react-native-paper';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { WebView } from 'react-native-webview';

type Message = {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: Date;
};

type AppRoute = '/dashboard' | '/reminders' | '/talk-to-ai' | '/profile';

export default function TalkToAIScreen() {
  const theme = useTheme();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Good Morning!',
      sender: 'ai',
      timestamp: new Date(),
    },
    {
      id: '2',
      text: 'How are you feeling today?',
      sender: 'ai',
      timestamp: new Date(),
    },
    {
      id: '3',
      text: "Dr. I don't feel well, I have a cough and a fever.",
      sender: 'user',
      timestamp: new Date(),
    },
    {
      id: '4',
      text: 'Can you send me a photo of medicine?',
      sender: 'ai',
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isCasting, setIsCasting] = useState(false);
  const scrollViewRef = useRef<ScrollView>(null);

  const handleSend = () => {
    if (inputText.trim()) {
      const newMessage: Message = {
        id: Date.now().toString(),
        text: inputText.trim(),
        sender: 'user',
        timestamp: new Date(),
      };
      setMessages([...messages, newMessage]);
      setInputText('');

      // Simulate AI response
      setTimeout(() => {
        const aiResponse: Message = {
          id: (Date.now() + 1).toString(),
          text: 'I understand. Let me help you with that...',
          sender: 'ai',
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, aiResponse]);
      }, 1000);
    }
  };

  const toggleRecording = () => {
    setIsRecording(!isRecording);
    // TODO: Implement actual voice recording
  };

  const toggleCasting = () => {
    setIsCasting(!isCasting);
    // TODO: Implement actual TV casting
  };

  const handleNavigation = (route: AppRoute) => {
    router.push(route as any); // TODO: Update route types when proper type definitions are available
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
    <SafeAreaView style={styles.container}>
      <StatusBar style="light" />

      {/* Header */}
      <Surface style={styles.header} elevation={2}>
        <View style={styles.headerContent}>
          <IconButton icon="arrow-left" size={24} onPress={() => router.back()} />
          <View style={styles.headerCenter}>
            <Text variant="titleMedium" style={styles.headerTitle}>Your Health Hub</Text>
            <View style={styles.aiStatus}>
              <View style={styles.statusDot} />
              <Text variant="bodySmall" style={styles.statusText}>AI Assistant Active</Text>
            </View>
          </View>
          <IconButton icon="dots-vertical" size={24} />
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
              styles.messageRow,
              message.sender === 'user' ? styles.userMessageRow : null,
            ]}
          >
            {message.sender === 'ai' && (
              <Avatar.Image
                size={40}
                source={require('../assets/ai-avatar.png')}
                style={styles.avatar}
              />
            )}
            <View
              style={[
                styles.messageBubble,
                message.sender === 'user' ? styles.userBubble : styles.aiBubble,
              ]}
            >
              <Text style={message.sender === 'user' ? styles.userMessageText : styles.aiMessageText}>
                {message.text}
              </Text>
            </View>
          </View>
        ))}
      </ScrollView>

      {/* Input Area */}
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 88 : 0}
        style={styles.inputContainer}
      >
        <Surface style={styles.inputWrapper} elevation={4}>
          <IconButton icon="image" size={24} />
          <TextInput
            value={inputText}
            onChangeText={setInputText}
            placeholder="Type your message..."
            mode="flat"
            style={styles.input}
            right={<TextInput.Icon icon="send" onPress={handleSend} />}
          />
          <IconButton icon="microphone" size={24} />
        </Surface>
      </KeyboardAvoidingView>

      {/* Bottom Navigation */}
      <Card style={styles.bottomNav}>
        <Card.Content style={styles.bottomNavContent}>
          <IconButton 
            icon="home" 
            size={24}
            onPress={() => handleNavigation('/dashboard')}
          />
          <IconButton 
            icon="calendar" 
            size={24}
            onPress={() => handleNavigation('/reminders')}
          />
          <IconButton
            icon="robot"
            size={24}
            mode="contained"
            containerColor="#E8F5F1"
            iconColor="#00B383"
            onPress={() => handleNavigation('/talk-to-ai')}
          />
          <IconButton 
            icon="cog" 
            size={24}
            onPress={() => handleNavigation('/profile')}
          />
        </Card.Content>
      </Card>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F6FA',
  },
  header: {
    backgroundColor: '#fff',
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 8,
  },
  headerCenter: {
    alignItems: 'center',
  },
  headerTitle: {
    fontWeight: 'bold',
  },
  aiStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#4CAF50',
  },
  statusText: {
    color: '#666',
  },
  modelContainer: {
    height: 300,
    backgroundColor: '#000',
  },
  webview: {
    flex: 1,
  },
  messagesContainer: {
    flex: 1,
  },
  messagesContent: {
    padding: 16,
    gap: 16,
  },
  messageRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 8,
  },
  userMessageRow: {
    flexDirection: 'row-reverse',
  },
  avatar: {
    backgroundColor: '#E8F5F1',
  },
  messageBubble: {
    maxWidth: '75%',
    padding: 12,
    borderRadius: 16,
  },
  userBubble: {
    backgroundColor: '#2D68FF',
    borderBottomRightRadius: 4,
  },
  aiBubble: {
    backgroundColor: '#fff',
    borderBottomLeftRadius: 4,
  },
  userMessageText: {
    color: '#fff',
  },
  aiMessageText: {
    color: '#000',
  },
  inputContainer: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: '#fff',
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    borderRadius: 24,
    backgroundColor: '#fff',
  },
  input: {
    flex: 1,
    backgroundColor: 'transparent',
  },
  bottomNav: {
    borderRadius: 0,
    elevation: 8,
  },
  bottomNavContent: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingHorizontal: 0,
  },
}); 