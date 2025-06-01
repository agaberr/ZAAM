import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Base API URL (should match your server configuration)
const API_URL = 'http://localhost:5003/api';

// User statistics interface
export interface UserStats {
  daysActive: number;
  aiInteractions: number;
  medicationAdherence: number; // percentage
  reminderCompletion: number; // percentage
  registrationDate?: string;
  lastActive?: string;
}

// Get auth token from async storage
const getAuthToken = async (): Promise<string | null> => {
  try {
    return await AsyncStorage.getItem('authToken');
  } catch (error) {
    console.error('Error getting auth token:', error);
    return null;
  }
};

// Create axios instance with authorization headers
const createAuthAPI = async () => {
  try {
    const token = await getAuthToken();
    
    return axios.create({
      baseURL: API_URL,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': token ? `Bearer ${token}` : '',
      },
      withCredentials: true,
    });
  } catch (error) {
    console.error('Error creating auth API:', error);
    return axios.create({
      baseURL: API_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }
};

export const userStatsService = {
  // Get user statistics
  getStats: async (): Promise<UserStats> => {
    try {
      const api = await createAuthAPI();
      const response = await api.get('/user/stats');
      
      if (response.data && response.data.success) {
        return response.data.stats;
      } else {
        // Return calculated stats if backend doesn't have dedicated endpoint
        return await userStatsService.calculateStats();
      }
    } catch (error) {
      console.error('Error fetching user stats:', error);
      // Return calculated stats as fallback
      return await userStatsService.calculateStats();
    }
  },

  // Calculate stats from available data
  calculateStats: async (): Promise<UserStats> => {
    try {
      // Get AI interactions count from local storage
      const aiInteractionsStr = await AsyncStorage.getItem('aiInteractionsCount');
      const aiInteractions = parseInt(aiInteractionsStr || '0');

      // Get user data to calculate days active
      const api = await createAuthAPI();
      let daysActive = 1; // Default to 1 day
      let registrationDate = new Date().toISOString();

      try {
        // Try to get user ID from AsyncStorage (assuming it's stored during login)
        const userId = await AsyncStorage.getItem('userId');
        if (userId) {
          const userResponse = await api.get(`/users/${userId}`);
          if (userResponse.data && userResponse.data.created_at) {
            registrationDate = userResponse.data.created_at;
            const regDate = new Date(registrationDate);
            const today = new Date();
            const timeDiff = today.getTime() - regDate.getTime();
            daysActive = Math.max(1, Math.ceil(timeDiff / (1000 * 3600 * 24)));
          }
        }
      } catch (userError) {
        console.log('Could not fetch user data for days active calculation:', userError);
        // Try to calculate from stored registration date as fallback
        try {
          const storedRegDate = await AsyncStorage.getItem('userRegistrationDate');
          if (storedRegDate) {
            const regDate = new Date(storedRegDate);
            const today = new Date();
            const timeDiff = today.getTime() - regDate.getTime();
            daysActive = Math.max(1, Math.ceil(timeDiff / (1000 * 3600 * 24)));
          }
        } catch (fallbackError) {
          console.log('Could not get registration date from storage:', fallbackError);
        }
      }

      // Calculate medication and reminder stats
      let medicationAdherence = 0;
      let reminderCompletion = 0;

      try {
        // Import reminderService dynamically to avoid circular dependency
        const { reminderService } = await import('./reminderService');
        const reminderStats = await reminderService.getStatistics();
        
        // Calculate medication adherence based on medication-type reminders
        const totalMedications = reminderStats.byType.medication;
        if (totalMedications > 0) {
          // For now, use completion rate as medication adherence
          // In a real app, you'd track medication-specific completion
          medicationAdherence = reminderStats.completionRate;
        }

        reminderCompletion = reminderStats.completionRate;
      } catch (reminderError) {
        console.error('Error getting reminder stats for user stats:', reminderError);
      }

      return {
        daysActive,
        aiInteractions,
        medicationAdherence,
        reminderCompletion,
        registrationDate,
        lastActive: new Date().toISOString(),
      };
    } catch (error) {
      console.error('Error calculating user stats:', error);
      // Return default stats
      return {
        daysActive: 1,
        aiInteractions: 0,
        medicationAdherence: 0,
        reminderCompletion: 0,
      };
    }
  },

  // Increment AI interactions count
  incrementAIInteractions: async (): Promise<void> => {
    try {
      // Get current count
      const currentCountStr = await AsyncStorage.getItem('aiInteractionsCount');
      const currentCount = parseInt(currentCountStr || '0');
      
      // Increment and save
      const newCount = currentCount + 1;
      await AsyncStorage.setItem('aiInteractionsCount', newCount.toString());

      console.log(`AI interactions incremented to: ${newCount}`);

      // Note: AI interactions are tracked locally only
      // AI requests go to /api/ai/process endpoint instead
    } catch (error) {
      console.error('Error incrementing AI interactions:', error);
    }
  },

  // Update last active timestamp
  updateLastActive: async (): Promise<void> => {
    try {
      await AsyncStorage.setItem('lastActiveTimestamp', new Date().toISOString());
      
      // Try to sync with backend if available
      try {
        const api = await createAuthAPI();
        await api.post('/user/stats/activity');
      } catch (backendError: any) {
        console.log('Could not sync activity with backend:', backendError.message);
      }
    } catch (error) {
      console.error('Error updating last active:', error);
    }
  },

  // Save user stats to backend
  saveStats: async (stats: Partial<UserStats>): Promise<void> => {
    try {
      const api = await createAuthAPI();
      await api.post('/user/stats', stats);
    } catch (error) {
      console.error('Error saving user stats:', error);
      // Store locally as fallback
      try {
        await AsyncStorage.setItem('userStats', JSON.stringify(stats));
      } catch (localError) {
        console.error('Error saving stats locally:', localError);
      }
    }
  },

  // Reset AI interactions count (useful for testing)
  resetAIInteractions: async (): Promise<void> => {
    try {
      await AsyncStorage.setItem('aiInteractionsCount', '0');
      console.log('AI interactions count reset to 0');
    } catch (error) {
      console.error('Error resetting AI interactions:', error);
    }
  },
}; 