import { Dimensions, ViewStyle } from 'react-native';

export const isDesktop = () => {
  const { width } = Dimensions.get('window');
  return width >= 768; // Standard tablet/desktop breakpoint
};

export const useResponsiveStyles = () => {
  const { width } = Dimensions.get('window');
  const isDesktopView = width >= 768;

  return {
    isDesktopView,
    containerStyle: {
      height: '100%',
      backgroundColor: '#2c1f16',
      position: 'relative',
      overflow: 'hidden',
    } as ViewStyle,
    chatContainerStyle: {
      position: 'absolute',
      bottom: 0,
      width: isDesktopView ? '0%' : '100%', // Hide on desktop, full width on mobile
      height: isDesktopView ? '0%' : '40%', // Hide on desktop, 40% height on mobile
      zIndex: 1,
      display: isDesktopView ? 'none' : 'flex',
      alignItems: 'center' as const,
      justifyContent: 'center' as const,
      backgroundColor: 'rgba(0, 0, 0, 0.7)',
    } as ViewStyle,
    canvasStyle: {
      height: isDesktopView ? '100%' : '60%', // Full height on desktop, 60% on mobile
    } as ViewStyle,
    voiceButtonStyle: {
      position: 'absolute',
      bottom: 20,
      right: 20,
      zIndex: 2,
    } as ViewStyle,
  };
}; 