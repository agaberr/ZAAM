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
  private retryCount = 0;
  private maxRetries = 3;
  private retryTimeout: NodeJS.Timeout | null = null;

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
        return;
      }

      // Web Speech API for web platform
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      
      if (SpeechRecognition) {
        this.recognition = new SpeechRecognition();
        
        // Configure speech recognition for better stability
        this.recognition.continuous = false;
        this.recognition.interimResults = false;
        this.recognition.lang = 'en-US';
        this.recognition.maxAlternatives = 1;
        
        // Add timeout to prevent hanging
        if (this.recognition.serviceURI !== undefined) {
          // Some browsers support serviceURI configuration
          this.recognition.serviceURI = 'wss://www.google.com/speech-api/full-duplex/v1/up';
        }

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
          this.retryCount = 0; // Reset retry count on successful result
          this.callbacks.onSpeechResult?.(transcript);
        };

        this.recognition.onerror = (event: any) => {
          console.error('Speech recognition error:', event.error, event);
          
          // IMPORTANT: Reset listening state on any error
          this.isListening = false;
          
          // Handle network errors with improved handling
          if (event.error === 'network') {
            this.handleNetworkError();
            return;
          }
          
          let errorMessage = event.error;
          
          // Provide more helpful error messages
          switch (event.error) {
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
            case 'aborted':
              // Don't show error for user-initiated aborts
              return;
            case 'bad-grammar':
              errorMessage = 'Speech recognition configuration error. Please try again.';
              break;
            default:
              errorMessage = `Speech recognition error: ${event.error}. Please try again.`;
          }
          
          this.callbacks.onSpeechError?.(errorMessage);
        };
      } else {
        console.warn('Speech Recognition API not supported in this browser');
      }
    }
  }

  private handleNetworkError() {
    // Stop any active recognition first
    this.forceStop();
    
    if (this.retryCount < this.maxRetries) {
      this.retryCount++;
      console.log(`Network error detected. Attempting retry ${this.retryCount}/${this.maxRetries}`);
      
      // Show user-friendly message for first retry
      if (this.retryCount === 1) {
        this.callbacks.onSpeechError?.('Connection issue detected. Please try again manually.');
      } else if (this.retryCount === 2) {
        this.callbacks.onSpeechError?.('Speech service is currently unavailable. Please try again.');
      } else {
        this.callbacks.onSpeechError?.('Unable to connect to speech service. Please check your internet connection.');
      }
      
      // Clear any existing retry timeout
      if (this.retryTimeout) {
        clearTimeout(this.retryTimeout);
        this.retryTimeout = null;
      }
      
      // Don't automatically retry - let user retry manually
      // This prevents the infinite retry loop you're experiencing
      this.retryCount = 0; // Reset for next manual attempt
    } else {
      this.callbacks.onSpeechError?.('Speech recognition service is currently unavailable. Please try again later.');
      this.retryCount = 0; // Reset for next time
    }
  }

  // Special retry method that doesn't reset retry count
  private async retryStartListening(): Promise<void> {
    try {
      if (Platform.OS === 'web') {
        // Clear any pending retry timeout to prevent conflicts
        if (this.retryTimeout) {
          clearTimeout(this.retryTimeout);
          this.retryTimeout = null;
        }
        
        // Force reset state before starting
        if (this.isListening) {
          console.warn('Force stopping previous recognition session for retry');
          this.forceStop();
        }

        if (this.recognition) {
          // Add a small delay to ensure previous session is fully closed
          await new Promise(resolve => setTimeout(resolve, 200));
          this.recognition.start();
        } else {
          throw new Error('Speech recognition not supported. Please use a modern browser with HTTPS.');
        }
      } else {
        throw new Error('Speech recognition not implemented for mobile yet');
      }
    } catch (error) {
      console.error('Error in retry start listening:', error);
      this.isListening = false;
      throw error; // Re-throw for retry logic
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
        // Clear any pending retry timeout to prevent conflicts
        if (this.retryTimeout) {
          clearTimeout(this.retryTimeout);
          this.retryTimeout = null;
        }
        
        // Force reset state before starting
        if (this.isListening) {
          console.warn('Force stopping previous recognition session');
          this.forceStop();
        }

        // Check microphone permissions first
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          try {
            await navigator.mediaDevices.getUserMedia({ audio: true });
          } catch (permissionError) {
            throw new Error('Microphone permission denied. Please allow microphone access and reload the page.');
          }
        }

        if (this.recognition) {
          // Add a small delay to ensure previous session is fully closed
          await new Promise(resolve => setTimeout(resolve, 100));
          this.recognition.start();
        } else {
          throw new Error('Speech recognition not supported. Please use a modern browser with HTTPS.');
        }
      } else {
        // For mobile platforms, you might want to use a different approach
        throw new Error('Speech recognition not implemented for mobile yet');
      }
    } catch (error) {
      console.error('Error starting speech recognition:', error);
      this.isListening = false; // Ensure state is reset on error
      this.callbacks.onSpeechError?.(error instanceof Error ? error.message : 'Unknown error');
      throw error; // Re-throw for retry logic
    }
  }

  // Stop speech recognition
  stopListening(): void {
    if (Platform.OS === 'web' && this.recognition) {
      try {
        if (this.isListening) {
          this.recognition.stop();
        }
      } catch (error) {
        console.error('Error stopping speech recognition:', error);
      } finally {
        // Always reset state
        this.isListening = false;
      }
    }
  }

  // Force stop and reset (for cleanup)
  private forceStop(): void {
    if (Platform.OS === 'web' && this.recognition) {
      try {
        this.recognition.abort(); // Use abort instead of stop for immediate termination
      } catch (error) {
        console.error('Error force stopping speech recognition:', error);
      } finally {
        this.isListening = false;
      }
    }
  }

  // Add a method to reset the service state
  reset(): void {
    this.forceStop();
    this.isListening = false;
    this.retryCount = 0;
    
    // Clear any pending retry timeout
    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout);
      this.retryTimeout = null;
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