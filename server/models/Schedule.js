const mongoose = require('mongoose');

const scheduleSchema = new mongoose.Schema({
  user_id: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  reminders: [{
    type: {
      type: String,
      required: true,
      enum: ['medication', 'hydration', 'appointment', 'exercise', 'other']
    },
    name: {
      type: String,
      required: true
    },
    dosage: {
      type: String
    },
    time: {
      type: String,
      required: true
    },
    recurrence: {
      type: String,
      required: true,
      enum: ['once', 'daily', 'weekly', 'monthly', 'every 2 hours', 'custom']
    },
    status: {
      type: String,
      default: 'pending',
      enum: ['pending', 'completed', 'missed']
    },
    last_taken: {
      type: Date
    }
  }],
  created_at: {
    type: Date,
    default: Date.now
  },
  updated_at: {
    type: Date,
    default: Date.now
  }
});

const Schedule = mongoose.model('Schedule', scheduleSchema);

module.exports = Schedule;
