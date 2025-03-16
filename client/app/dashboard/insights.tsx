import React, { useState } from 'react';
import { StyleSheet, View, ScrollView, Dimensions, TouchableOpacity } from 'react-native';
import { Text, Card, Button, Chip, IconButton, ProgressBar, SegmentedButtons } from 'react-native-paper';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { BarChart, LineChart, PieChart } from 'react-native-chart-kit';
import * as Sharing from 'expo-sharing';
import * as FileSystem from 'expo-file-system';
import BottomNavigation from '../components/BottomNavigation';

const { width } = Dimensions.get('window');

interface DataPoint {
  value: number;
  day: string;
}

interface MedicationData {
  name: string;
  adherence: number;
  color: string;
}

interface AIInteractionData {
  date: string;
  duration: number;
  topics: string[];
}

export default function InsightsScreen() {
  const router = useRouter();
  const [timeRange, setTimeRange] = useState('week');
  const [dataType, setDataType] = useState('glucose');
  
  // Sample glucose data
  const glucoseData: DataPoint[] = [
    { value: 140, day: 'Sun' },
    { value: 105, day: 'Mon' },
    { value: 180, day: 'Tue' },
    { value: 210, day: 'Wed' },
    { value: 150, day: 'Thu' },
    { value: 100, day: 'Fri' },
    { value: 120, day: 'Sat' },
  ];
  
  // Sample blood pressure data
  const bpData: DataPoint[] = [
    { value: 120, day: 'Sun' },
    { value: 125, day: 'Mon' },
    { value: 130, day: 'Tue' },
    { value: 135, day: 'Wed' },
    { value: 125, day: 'Thu' },
    { value: 120, day: 'Fri' },
    { value: 115, day: 'Sat' },
  ];
  
  // Sample medication adherence data
  const medicationData: MedicationData[] = [
    { name: 'Aspirin', adherence: 95, color: '#4285F4' },
    { name: 'Donepezil', adherence: 85, color: '#34C759' },
    { name: 'Memantine', adherence: 75, color: '#FF9500' },
  ];
  
  // Sample AI interaction data
  const aiInteractionData: AIInteractionData[] = [
    { date: 'Monday', duration: 15, topics: ['Medication', 'Family'] },
    { date: 'Tuesday', duration: 25, topics: ['Memory Exercise', 'Weather'] },
    { date: 'Wednesday', duration: 30, topics: ['Appointments', 'News'] },
    { date: 'Thursday', duration: 10, topics: ['Medication'] },
    { date: 'Friday', duration: 20, topics: ['Family Photos', 'Reminders'] },
    { date: 'Saturday', duration: 5, topics: ['Weather'] },
    { date: 'Sunday', duration: 18, topics: ['Memory Exercise', 'Medication'] },
  ];
  
  // Calculate statistics
  const getHighestValue = (data: DataPoint[]) => {
    return Math.max(...data.map(item => item.value));
  };
  
  const getAverageValue = (data: DataPoint[]) => {
    const sum = data.reduce((acc, item) => acc + item.value, 0);
    return Math.round(sum / data.length);
  };
  
  const getTotalAIInteractionTime = () => {
    return aiInteractionData.reduce((acc, item) => acc + item.duration, 0);
  };
  
  const getAverageAIInteractionTime = () => {
    return Math.round(getTotalAIInteractionTime() / aiInteractionData.length);
  };
  
  const getMostDiscussedTopics = () => {
    const topicCounts: Record<string, number> = {};
    
    aiInteractionData.forEach(interaction => {
      interaction.topics.forEach(topic => {
        topicCounts[topic] = (topicCounts[topic] || 0) + 1;
      });
    });
    
    return Object.entries(topicCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([topic]) => topic);
  };
  
  const getAverageMedicationAdherence = () => {
    const sum = medicationData.reduce((acc, item) => acc + item.adherence, 0);
    return Math.round(sum / medicationData.length);
  };
  
  // Chart configuration
  const chartConfig = {
    backgroundGradientFrom: '#fff',
    backgroundGradientTo: '#fff',
    color: (opacity = 1) => `rgba(66, 133, 244, ${opacity})`,
    strokeWidth: 2,
    barPercentage: 0.7,
    useShadowColorFromDataset: false,
    decimalPlaces: 0,
  };
  
  const barData = {
    labels: glucoseData.map(item => item.day),
    datasets: [
      {
        data: dataType === 'glucose' 
          ? glucoseData.map(item => item.value)
          : bpData.map(item => item.value),
      },
    ],
  };
  
  const pieData = medicationData.map(item => ({
    name: item.name,
    adherence: item.adherence,
    color: item.color,
    legendFontColor: '#7F7F7F',
    legendFontSize: 12,
  }));
  
  const lineData = {
    labels: aiInteractionData.map(item => item.date.substring(0, 3)),
    datasets: [
      {
        data: aiInteractionData.map(item => item.duration),
        color: (opacity = 1) => `rgba(66, 133, 244, ${opacity})`,
        strokeWidth: 2,
      },
    ],
  };
  
  const handleDownloadReport = async () => {
    try {
      // In a real app, this would generate a PDF report
      const dummyPdfPath = FileSystem.documentDirectory + 'health_report.pdf';
      
      // For demo purposes, we'll just create a text file
      await FileSystem.writeAsStringAsync(
        dummyPdfPath,
        'This is a sample health report for demonstration purposes.',
        { encoding: FileSystem.EncodingType.UTF8 }
      );
      
      // Share the file
      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(dummyPdfPath);
      }
    } catch (error) {
      console.error('Error sharing report:', error);
    }
  };
  
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Insights</Text>
        <IconButton
          icon="bell-outline"
          size={24}
          onPress={() => {}}
        />
      </View>
      
      <ScrollView style={styles.scrollView}>
        {/* Time range selector */}
        <View style={styles.timeRangeContainer}>
          <SegmentedButtons
            value={timeRange}
            onValueChange={setTimeRange}
            buttons={[
              { value: 'week', label: 'Week' },
              { value: 'month', label: 'Month' },
              { value: 'year', label: 'Year' },
            ]}
            style={styles.segmentedButtons}
          />
        </View>
        
        {/* Main chart card */}
        <Card style={styles.chartCard}>
          <Card.Content>
            <View style={styles.chartHeader}>
              <Text style={styles.chartTitle}>
                {timeRange === 'week' ? 'Weekly' : timeRange === 'month' ? 'Monthly' : 'Yearly'} average, 
                {dataType === 'glucose' ? ' glucose level' : ' blood pressure'}
              </Text>
              <View style={styles.dataTypeSelector}>
                <Chip 
                  selected={dataType === 'glucose'} 
                  onPress={() => setDataType('glucose')}
                  style={styles.dataTypeChip}
                >
                  Glucose
                </Chip>
                <Chip 
                  selected={dataType === 'bp'} 
                  onPress={() => setDataType('bp')}
                  style={styles.dataTypeChip}
                >
                  BP
                </Chip>
              </View>
            </View>
            
            <View style={styles.chartContainer}>
              <BarChart
                data={barData}
                width={width - 60}
                height={220}
                chartConfig={chartConfig}
                style={styles.chart}
                showValuesOnTopOfBars
                fromZero
              />
              
              <View style={styles.thresholdLine}>
                <Text style={styles.thresholdText}>High</Text>
              </View>
            </View>
          </Card.Content>
        </Card>
        
        {/* Weekly summary */}
        <Text style={styles.sectionTitle}>Weekly summary</Text>
        <View style={styles.summaryContainer}>
          <Card style={styles.summaryCard}>
            <Card.Content>
              <Text style={styles.summaryLabel}>Highest Glucose</Text>
              <Text style={[styles.summaryValue, styles.highValue]}>
                {getHighestValue(glucoseData)} <Text style={styles.unit}>mg/dl</Text>
              </Text>
            </Card.Content>
          </Card>
          
          <Card style={styles.summaryCard}>
            <Card.Content>
              <Text style={styles.summaryLabel}>Avg Glucose</Text>
              <Text style={[styles.summaryValue, styles.avgValue]}>
                {getAverageValue(glucoseData)} <Text style={styles.unit}>mg/dl</Text>
              </Text>
            </Card.Content>
          </Card>
        </View>
        
        {/* Monthly summary */}
        <Text style={styles.sectionTitle}>Monthly summary</Text>
        <View style={styles.summaryContainer}>
          <Card style={styles.summaryCard}>
            <Card.Content>
              <Text style={styles.summaryLabel}>Highest Glucose</Text>
              <Text style={[styles.summaryValue, styles.highValue]}>
                216 <Text style={styles.unit}>mg/dl</Text>
              </Text>
            </Card.Content>
          </Card>
          
          <Card style={styles.summaryCard}>
            <Card.Content>
              <Text style={styles.summaryLabel}>Avg Glucose</Text>
              <Text style={[styles.summaryValue, styles.avgValue]}>
                144 <Text style={styles.unit}>mg/dl</Text>
              </Text>
            </Card.Content>
          </Card>
        </View>
        
        {/* Medication Adherence (Creative Addition) */}
        <Text style={styles.sectionTitle}>Medication Adherence</Text>
        <Card style={styles.medicationCard}>
          <Card.Content>
            <View style={styles.medicationHeader}>
              <Text style={styles.medicationTitle}>Overall Adherence</Text>
              <Chip mode="outlined" style={styles.adherenceChip}>
                {getAverageMedicationAdherence()}%
              </Chip>
            </View>
            
            {medicationData.map((medication, index) => (
              <View key={index} style={styles.medicationItem}>
                <View style={styles.medicationNameContainer}>
                  <View 
                    style={[styles.medicationColorDot, { backgroundColor: medication.color }]} 
                  />
                  <Text style={styles.medicationName}>{medication.name}</Text>
                </View>
                <View style={styles.medicationProgressContainer}>
                  <ProgressBar 
                    progress={medication.adherence / 100} 
                    color={medication.color}
                    style={styles.medicationProgress} 
                  />
                  <Text style={styles.medicationPercentage}>{medication.adherence}%</Text>
                </View>
              </View>
            ))}
          </Card.Content>
        </Card>
        
        {/* AI Interaction Analytics (Creative Addition) */}
        <Text style={styles.sectionTitle}>AI Companion Interactions</Text>
        <Card style={styles.aiCard}>
          <Card.Content>
            <View style={styles.aiStatsContainer}>
              <View style={styles.aiStatItem}>
                <Text style={styles.aiStatValue}>{getTotalAIInteractionTime()}</Text>
                <Text style={styles.aiStatLabel}>Total Minutes</Text>
              </View>
              
              <View style={styles.aiStatDivider} />
              
              <View style={styles.aiStatItem}>
                <Text style={styles.aiStatValue}>{getAverageAIInteractionTime()}</Text>
                <Text style={styles.aiStatLabel}>Avg Minutes/Day</Text>
              </View>
              
              <View style={styles.aiStatDivider} />
              
              <View style={styles.aiStatItem}>
                <Text style={styles.aiStatValue}>{aiInteractionData.length}</Text>
                <Text style={styles.aiStatLabel}>Conversations</Text>
              </View>
            </View>
            
            <Text style={styles.aiChartTitle}>Daily Interaction Time (minutes)</Text>
            <LineChart
              data={lineData}
              width={width - 60}
              height={180}
              chartConfig={{
                ...chartConfig,
                color: (opacity = 1) => `rgba(66, 133, 244, ${opacity})`,
                backgroundGradientFrom: '#f5f7fa',
                backgroundGradientTo: '#f5f7fa',
              }}
              bezier
              style={styles.aiChart}
            />
            
            <Text style={styles.aiTopicsTitle}>Most Discussed Topics</Text>
            <View style={styles.aiTopicsContainer}>
              {getMostDiscussedTopics().map((topic, index) => (
                <Chip key={index} style={styles.topicChip}>
                  {topic}
                </Chip>
              ))}
            </View>
          </Card.Content>
        </Card>
        
        {/* Cognitive Health Score (Creative Addition) */}
        <Text style={styles.sectionTitle}>Cognitive Health Score</Text>
        <Card style={styles.cognitiveCard}>
          <Card.Content>
            <View style={styles.cognitiveScoreContainer}>
              <View style={styles.cognitiveScoreCircle}>
                <Text style={styles.cognitiveScoreValue}>78</Text>
                <Text style={styles.cognitiveScoreLabel}>Good</Text>
              </View>
              
              <View style={styles.cognitiveDetailsContainer}>
                <Text style={styles.cognitiveDetailsTitle}>
                  Based on AI interactions and activity patterns
                </Text>
                <Text style={styles.cognitiveDetailsText}>
                  Your cognitive engagement has improved by 12% in the last month.
                </Text>
                <TouchableOpacity style={styles.cognitiveTipsButton}>
                  <Text style={styles.cognitiveTipsText}>View Recommendations</Text>
                  <Ionicons name="chevron-forward" size={16} color="#4285F4" />
                </TouchableOpacity>
              </View>
            </View>
            
            <View style={styles.cognitiveFactorsContainer}>
              <Text style={styles.cognitiveFactorsTitle}>Contributing Factors</Text>
              
              <View style={styles.cognitiveFactor}>
                <Text style={styles.cognitiveFactorLabel}>Memory Exercises</Text>
                <View style={styles.cognitiveFactorValueContainer}>
                  <Text style={[styles.cognitiveFactorValue, styles.positiveValue]}>+15%</Text>
                  <Ionicons name="arrow-up" size={16} color="#34C759" />
                </View>
              </View>
              
              <View style={styles.cognitiveFactor}>
                <Text style={styles.cognitiveFactorLabel}>Social Interactions</Text>
                <View style={styles.cognitiveFactorValueContainer}>
                  <Text style={[styles.cognitiveFactorValue, styles.positiveValue]}>+8%</Text>
                  <Ionicons name="arrow-up" size={16} color="#34C759" />
                </View>
              </View>
              
              <View style={styles.cognitiveFactor}>
                <Text style={styles.cognitiveFactorLabel}>Sleep Quality</Text>
                <View style={styles.cognitiveFactorValueContainer}>
                  <Text style={[styles.cognitiveFactorValue, styles.negativeValue]}>-3%</Text>
                  <Ionicons name="arrow-down" size={16} color="#FF3B30" />
                </View>
              </View>
            </View>
          </Card.Content>
        </Card>
        
        {/* Share report button */}
        <Text style={styles.sectionTitle}>Share your report</Text>
        <Button
          mode="contained"
          icon="download"
          onPress={handleDownloadReport}
          style={styles.downloadButton}
          contentStyle={styles.downloadButtonContent}
        >
          Download report
        </Button>
        
        {/* Bottom spacer */}
        <View style={styles.bottomSpacer} />
      </ScrollView>
      
      {/* Fixed Bottom Navigation */}
      <BottomNavigation />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F7FA',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 10,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#E1E1E1',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  scrollView: {
    flex: 1,
    padding: 16,
  },
  timeRangeContainer: {
    marginBottom: 16,
  },
  segmentedButtons: {
    backgroundColor: 'white',
  },
  chartCard: {
    marginBottom: 20,
    borderRadius: 16,
    elevation: 2,
  },
  chartHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  chartTitle: {
    fontSize: 16,
    color: '#666',
    flex: 1,
  },
  dataTypeSelector: {
    flexDirection: 'row',
  },
  dataTypeChip: {
    marginLeft: 8,
  },
  chartContainer: {
    position: 'relative',
    alignItems: 'center',
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  thresholdLine: {
    position: 'absolute',
    top: 40,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
  },
  thresholdText: {
    color: '#FF3B30',
    fontSize: 12,
    marginRight: 5,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
    marginTop: 8,
  },
  summaryContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  summaryCard: {
    width: '48%',
    borderRadius: 16,
    backgroundColor: '#F0F5FF',
  },
  summaryLabel: {
    fontSize: 16,
    color: '#666',
    marginBottom: 8,
  },
  summaryValue: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  highValue: {
    color: '#FF9500',
  },
  avgValue: {
    color: '#34C759',
  },
  unit: {
    fontSize: 14,
    fontWeight: 'normal',
    color: '#666',
  },
  medicationCard: {
    marginBottom: 20,
    borderRadius: 16,
  },
  medicationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  medicationTitle: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  adherenceChip: {
    backgroundColor: '#E8F1FF',
  },
  medicationItem: {
    marginBottom: 12,
  },
  medicationNameContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 5,
  },
  medicationColorDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 8,
  },
  medicationName: {
    fontSize: 14,
  },
  medicationProgressContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  medicationProgress: {
    flex: 1,
    height: 8,
    borderRadius: 4,
  },
  medicationPercentage: {
    marginLeft: 8,
    fontSize: 14,
    fontWeight: '500',
    width: 40,
    textAlign: 'right',
  },
  aiCard: {
    marginBottom: 20,
    borderRadius: 16,
  },
  aiStatsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#E1E1E1',
  },
  aiStatItem: {
    flex: 1,
    alignItems: 'center',
  },
  aiStatValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#4285F4',
    marginBottom: 5,
  },
  aiStatLabel: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
  aiStatDivider: {
    width: 1,
    height: 40,
    backgroundColor: '#E1E1E1',
  },
  aiChartTitle: {
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 10,
  },
  aiChart: {
    borderRadius: 16,
    marginBottom: 15,
  },
  aiTopicsTitle: {
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 10,
  },
  aiTopicsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  topicChip: {
    backgroundColor: '#E8F1FF',
    marginBottom: 5,
  },
  cognitiveCard: {
    marginBottom: 20,
    borderRadius: 16,
  },
  cognitiveScoreContainer: {
    flexDirection: 'row',
    marginBottom: 20,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#E1E1E1',
  },
  cognitiveScoreCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#4285F4',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 15,
  },
  cognitiveScoreValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
  },
  cognitiveScoreLabel: {
    fontSize: 12,
    color: 'white',
  },
  cognitiveDetailsContainer: {
    flex: 1,
    justifyContent: 'center',
  },
  cognitiveDetailsTitle: {
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 5,
  },
  cognitiveDetailsText: {
    fontSize: 12,
    color: '#666',
    marginBottom: 8,
  },
  cognitiveTipsButton: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  cognitiveTipsText: {
    fontSize: 14,
    color: '#4285F4',
    marginRight: 5,
  },
  cognitiveFactorsContainer: {
    marginTop: 5,
  },
  cognitiveFactorsTitle: {
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 10,
  },
  cognitiveFactor: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  cognitiveFactorLabel: {
    fontSize: 14,
  },
  cognitiveFactorValueContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  cognitiveFactorValue: {
    fontSize: 14,
    fontWeight: '500',
    marginRight: 5,
  },
  positiveValue: {
    color: '#34C759',
  },
  negativeValue: {
    color: '#FF3B30',
  },
  downloadButton: {
    marginBottom: 20,
    borderRadius: 12,
    backgroundColor: '#4285F4',
  },
  downloadButtonContent: {
    height: 50,
  },
  bottomSpacer: {
    height: 80,
  },
}); 