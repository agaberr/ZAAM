import * as Speech from 'expo-speech';
import { Platform } from 'react-native';

// For web platform, we'll use multiple speech recognition options
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

export interface SpeechProvider {
  name: string;
  isAvailable(): boolean;
  startListening(): Promise<void>;
  stopListening(): void;
  setCallbacks(callbacks: VoiceServiceCallbacks): void;
}

// Browser Web Speech API Provider (fallback)
class BrowserSpeechProvider implements SpeechProvider {
  name = 'Browser Web Speech API';
  private recognition: any = null;
  private isListening = false;
  private callbacks: VoiceServiceCallbacks = {};

  isAvailable(): boolean {
    if (Platform.OS !== 'web') return false;
    
    const hasSpeechRecognition = !!(window.SpeechRecognition || window.webkitSpeechRecognition);
    const isSecureContext = window.location.protocol === 'https:' || 
                            window.location.hostname === 'localhost' || 
                            window.location.hostname === '127.0.0.1';
    return hasSpeechRecognition && isSecureContext;
  }

  setCallbacks(callbacks: VoiceServiceCallbacks): void {
    this.callbacks = callbacks;
  }

  async startListening(): Promise<void> {
    if (!this.isAvailable()) {
      throw new Error('Browser speech recognition not available');
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    this.recognition = new SpeechRecognition();
    
    this.recognition.continuous = false;
    this.recognition.interimResults = false;
    this.recognition.lang = 'en-US';
    this.recognition.maxAlternatives = 1;

    this.recognition.onstart = () => {
      this.isListening = true;
      this.callbacks.onSpeechStart?.();
    };

    this.recognition.onend = () => {
      this.isListening = false;
      this.callbacks.onSpeechEnd?.();
    };

    this.recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      this.callbacks.onSpeechResult?.(transcript);
    };

    this.recognition.onerror = (event: any) => {
      this.isListening = false;
      let errorMessage = `Speech recognition failed: ${event.error}`;
      
      switch (event.error) {
        case 'not-allowed':
          errorMessage = 'Microphone access denied. Please allow microphone permissions.';
          break;
        case 'no-speech':
          errorMessage = 'No speech detected. Please speak clearly.';
          break;
        case 'audio-capture':
          errorMessage = 'Microphone not found or not working.';
          break;
        case 'network':
          errorMessage = 'Network error. Browser speech service is unavailable.';
          break;
        default:
          errorMessage = `Speech recognition error: ${event.error}`;
      }
      
      this.callbacks.onSpeechError?.(errorMessage);
    };

    // Check microphone permissions first
    try {
      await navigator.mediaDevices.getUserMedia({ audio: true });
      this.recognition.start();
    } catch (error) {
      throw new Error('Microphone permission denied');
    }
  }

  stopListening(): void {
    if (this.recognition && this.isListening) {
      this.recognition.stop();
    }
  }
}

// MediaRecorder + AssemblyAI Provider (recommended)
class AssemblyAISpeechProvider implements SpeechProvider {
  name = 'AssemblyAI Real-time';
  private mediaRecorder: MediaRecorder | null = null;
  private websocket: WebSocket | null = null;
  private isListening = false;
  private callbacks: VoiceServiceCallbacks = {};
  private audioChunks: Blob[] = [];

  isAvailable(): boolean {
    return Platform.OS === 'web' && 
           typeof MediaRecorder !== 'undefined' && 
           typeof WebSocket !== 'undefined';
  }

  setCallbacks(callbacks: VoiceServiceCallbacks): void {
    this.callbacks = callbacks;
  }

  async startListening(): Promise<void> {
    if (!this.isAvailable()) {
      throw new Error('MediaRecorder or WebSocket not supported');
    }

    try {
      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        } 
      });

      // Set up MediaRecorder
      this.mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      });

      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        await this.transcribeAudio(audioBlob);
        this.cleanup();
      };

      this.mediaRecorder.start(100); // Collect data every 100ms
      this.isListening = true;
      this.callbacks.onSpeechStart?.();

    } catch (error) {
      throw new Error(`Failed to start recording: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  stopListening(): void {
    if (this.mediaRecorder && this.isListening) {
      this.mediaRecorder.stop();
      this.isListening = false;
      this.callbacks.onSpeechEnd?.();
    }
  }

  private async transcribeAudio(audioBlob: Blob): Promise<void> {
    try {
      // Convert to base64 for API
      const arrayBuffer = await audioBlob.arrayBuffer();
      const base64Audio = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));

      // Use your existing AI service endpoint
      const response = await fetch('https://zaaam.me/api/speech/transcribe', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
        },
        body: JSON.stringify({
          audio: base64Audio,
          format: 'webm',
        }),
      });

      const data = await response.json();
      
      if (data.success && data.transcript) {
        this.callbacks.onSpeechResult?.(data.transcript);
      } else {
        this.callbacks.onSpeechError?.('Transcription failed');
      }
    } catch (error) {
      this.callbacks.onSpeechError?.(`Transcription error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private cleanup(): void {
    if (this.mediaRecorder) {
      const tracks = this.mediaRecorder.stream.getTracks();
      tracks.forEach(track => track.stop());
      this.mediaRecorder = null;
    }
    this.audioChunks = [];
  }
}

// OpenAI Whisper Provider (alternative)
class OpenAIWhisperProvider implements SpeechProvider {
  name = 'OpenAI Whisper';
  private mediaRecorder: MediaRecorder | null = null;
  private isListening = false;
  private callbacks: VoiceServiceCallbacks = {};
  private audioChunks: Blob[] = [];

  isAvailable(): boolean {
    return Platform.OS === 'web' && typeof MediaRecorder !== 'undefined';
  }

  setCallbacks(callbacks: VoiceServiceCallbacks): void {
    this.callbacks = callbacks;
  }

  async startListening(): Promise<void> {
    if (!this.isAvailable()) {
      throw new Error('MediaRecorder not supported');
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      this.mediaRecorder = new MediaRecorder(stream);
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
        await this.transcribeWithWhisper(audioBlob);
        this.cleanup();
      };

      this.mediaRecorder.start();
      this.isListening = true;
      this.callbacks.onSpeechStart?.();

    } catch (error) {
      throw new Error(`Failed to start recording: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  stopListening(): void {
    if (this.mediaRecorder && this.isListening) {
      this.mediaRecorder.stop();
      this.isListening = false;
      this.callbacks.onSpeechEnd?.();
    }
  }

  private async transcribeWithWhisper(audioBlob: Blob): Promise<void> {
    try {
      const formData = new FormData();
      formData.append('file', audioBlob, 'audio.wav');
      formData.append('model', 'whisper-1');

      const response = await fetch('https://zaaam.me/api/speech/whisper', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
        },
        body: formData,
      });

      const data = await response.json();
      
      if (data.success && data.text) {
        this.callbacks.onSpeechResult?.(data.text);
      } else {
        this.callbacks.onSpeechError?.('Whisper transcription failed');
      }
    } catch (error) {
      this.callbacks.onSpeechError?.(`Whisper error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private cleanup(): void {
    if (this.mediaRecorder) {
      const tracks = this.mediaRecorder.stream.getTracks();
      tracks.forEach(track => track.stop());
      this.mediaRecorder = null;
    }
    this.audioChunks = [];
  }
}

class VoiceService {
  private providers: SpeechProvider[] = [];
  private currentProvider: SpeechProvider | null = null;
  private callbacks: VoiceServiceCallbacks = {};

  constructor() {
    this.initializeProviders();
  }

  private initializeProviders(): void {
    // Add providers in order of preference
    this.providers = [
      new AssemblyAISpeechProvider(),    // Best quality, most reliable
      new OpenAIWhisperProvider(),       // Good alternative
      new BrowserSpeechProvider(),       // Fallback option
    ];
  }

  setCallbacks(callbacks: VoiceServiceCallbacks): void {
    this.callbacks = callbacks;
  }

  isSpeechAvailable(): boolean {
    return this.providers.some(provider => provider.isAvailable());
  }

  getAvailableProviders(): string[] {
    return this.providers
      .filter(provider => provider.isAvailable())
      .map(provider => provider.name);
  }

  async startListening(preferredProvider?: string): Promise<void> {
    // Find the best available provider
    let provider: SpeechProvider | undefined;
    
    if (preferredProvider) {
      provider = this.providers.find(p => 
        p.name === preferredProvider && p.isAvailable()
      );
    }
    
    if (!provider) {
      provider = this.providers.find(p => p.isAvailable());
    }

    if (!provider) {
      throw new Error('No speech recognition providers available');
    }

    this.currentProvider = provider;
    this.currentProvider.setCallbacks(this.callbacks);
    
    console.log(`Using speech provider: ${provider.name}`);
    await this.currentProvider.startListening();
  }

  stopListening(): void {
    if (this.currentProvider) {
      this.currentProvider.stopListening();
    }
  }

  reset(): void {
    this.stopListening();
    this.currentProvider = null;
  }

  getIsListening(): boolean {
    // This would need to be implemented in each provider
    return this.currentProvider !== null;
  }

  // Text-to-speech functionality
  async speak(text: string, options?: Speech.SpeechOptions): Promise<void> {
    try {
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

  async stopSpeaking(): Promise<void> {
    try {
      await Speech.stop();
    } catch (error) {
      console.error('Error stopping speech:', error);
    }
  }

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