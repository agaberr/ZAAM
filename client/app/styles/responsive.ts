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
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      width: isDesktopView ? '0%' : '100%', // Hide on desktop, full width on mobile
      height: isDesktopView ? '0%' : '100%', // Hide on desktop, full height on mobile
      zIndex: 1,
      display: isDesktopView ? 'none' : 'flex',
      backgroundColor: '#FFFFFF',
    } as ViewStyle,
    canvasStyle: {
      display: isDesktopView ? 'flex' : 'none', // Show on desktop, hide on mobile
      height: '100%',
    } as ViewStyle,
    voiceButtonStyle: {
      position: 'absolute',
      bottom: 20,
      right: 20,
      zIndex: 2,
      display: isDesktopView ? 'flex' : 'none', // Only show on desktop
    } as ViewStyle,
  };
}; 