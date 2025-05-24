import * as Speech from 'expo-speech';
import { Platform } from 'react-native';

// For web platform, we'll use the Web Speech API
declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}

export interface VoiceServiceCallbacks {
  onSpeechStart?: () => void;
  onSpeechEnd?: () => void;
  onSpeechResult?: (text: string) => void;
  onSpeechError?: (error: string) => void;
}

class VoiceService {
  private recognition: any = null;
  private isListening = false;
  private callbacks: VoiceServiceCallbacks = {};

  constructor() {
    this.initializeSpeechRecognition();
  }

  private initializeSpeechRecognition() {
    if (Platform.OS === 'web') {
      // Check if we're on HTTPS or localhost
      const isSecureContext = window.location.protocol === 'https:' || 
                              window.location.hostname === 'localhost' || 
                              window.location.hostname === '127.0.0.1';
      
      if (!isSecureContext) {
        console.warn('Speech Recognition requires HTTPS. Please use HTTPS or localhost.');
      }

      // Web Speech API for web platform
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      
      if (SpeechRecognition) {
        this.recognition = new SpeechRecognition();
        this.recognition.continuous = false;
        this.recognition.interimResults = false;
        this.recognition.lang = 'en-US';

        this.recognition.onstart = () => {
          this.isListening = true;
          this.callbacks.onSpeechStart?.();
          console.log('Speech recognition started');
        };

        this.recognition.onend = () => {
          this.isListening = false;
          this.callbacks.onSpeechEnd?.();
          console.log('Speech recognition ended');
        };

        this.recognition.onresult = (event: any) => {
          const transcript = event.results[0][0].transcript;
          console.log('Speech recognition result:', transcript);
          this.callbacks.onSpeechResult?.(transcript);
        };

        this.recognition.onerror = (event: any) => {
          console.error('Speech recognition error:', event.error, event);
          let errorMessage = event.error;
          
          // Provide more helpful error messages
          switch (event.error) {
            case 'network':
              errorMessage = 'Network error: Please ensure you have internet connection and try using HTTPS (https://localhost:8081)';
              break;
            case 'not-allowed':
              errorMessage = 'Microphone access denied. Please allow microphone permissions and reload the page.';
              break;
            case 'no-speech':
              errorMessage = 'No speech detected. Please try speaking louder or closer to the microphone.';
              break;
            case 'audio-capture':
              errorMessage = 'Microphone not found or not working. Please check your microphone.';
              break;
            case 'service-not-allowed':
              errorMessage = 'Speech service not allowed. Please enable microphone permissions.';
              break;
            default:
              errorMessage = `Speech recognition error: ${event.error}. Try using HTTPS or check your microphone.`;
          }
          
          this.callbacks.onSpeechError?.(errorMessage);
        };
      } else {
        console.warn('Speech Recognition API not supported in this browser');
      }
    }
  }

  // Set callbacks for speech recognition events
  setCallbacks(callbacks: VoiceServiceCallbacks) {
    this.callbacks = { ...this.callbacks, ...callbacks };
  }

  // Start speech recognition
  async startListening(): Promise<void> {
    try {
      if (Platform.OS === 'web') {
        // Check microphone permissions first
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          try {
            await navigator.mediaDevices.getUserMedia({ audio: true });
          } catch (permissionError) {
            throw new Error('Microphone permission denied. Please allow microphone access and reload the page.');
          }
        }

        if (this.recognition && !this.isListening) {
          this.recognition.start();
        } else if (!this.recognition) {
          throw new Error('Speech recognition not supported. Please use a modern browser with HTTPS.');
        } else {
          throw new Error('Already listening. Please wait for current recognition to complete.');
        }
      } else {
        // For mobile platforms, you might want to use a different approach
        throw new Error('Speech recognition not implemented for mobile yet');
      }
    } catch (error) {
      console.error('Error starting speech recognition:', error);
      this.callbacks.onSpeechError?.(error instanceof Error ? error.message : 'Unknown error');
    }
  }

  // Stop speech recognition
  stopListening(): void {
    if (Platform.OS === 'web' && this.recognition && this.isListening) {
      this.recognition.stop();
    }
  }

  // Check if currently listening
  getIsListening(): boolean {
    return this.isListening;
  }

  // Text-to-speech functionality
  async speak(text: string, options?: Speech.SpeechOptions): Promise<void> {
    try {
      // Stop any current speech
      await this.stopSpeaking();
      
      const speechOptions: Speech.SpeechOptions = {
        language: 'en-US',
        pitch: 1.0,
        rate: 0.8,
        ...options,
      };

      await Speech.speak(text, speechOptions);
    } catch (error) {
      console.error('Error in text-to-speech:', error);
      throw error;
    }
  }

  // Stop current speech
  async stopSpeaking(): Promise<void> {
    try {
      await Speech.stop();
    } catch (error) {
      console.error('Error stopping speech:', error);
    }
  }

  // Check if speech is available
  isSpeechAvailable(): boolean {
    if (Platform.OS === 'web') {
      const hasSpeechRecognition = !!(window.SpeechRecognition || window.webkitSpeechRecognition);
      const isSecureContext = window.location.protocol === 'https:' || 
                              window.location.hostname === 'localhost' || 
                              window.location.hostname === '127.0.0.1';
      return hasSpeechRecognition && isSecureContext;
    }
    return false; // For mobile, implement based on your needs
  }

  // Get available voices (for text-to-speech)
  async getAvailableVoices(): Promise<Speech.Voice[]> {
    try {
      return await Speech.getAvailableVoicesAsync();
    } catch (error) {
      console.error('Error getting available voices:', error);
      return [];
    }
  }
}

// Export singleton instance
export const voiceService = new VoiceService(); 