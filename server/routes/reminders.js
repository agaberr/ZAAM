const express = require('express');
const Schedule = require('../models/Schedule');
const auth = require('../middleware/auth');
const router = express.Router();

// Get all reminders for a user
router.get('/', auth, async (req, res) => {
  try {
    const schedule = await Schedule.findOne({ user_id: req.userId });
    if (!schedule) {
      return res.json({ reminders: [] });
    }
    res.json({ reminders: schedule.reminders });
  } catch (error) {
    console.error('Error fetching reminders:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Add a new reminder
router.post('/', auth, async (req, res) => {
  try {
    const reminderData = req.body;
    let schedule = await Schedule.findOne({ user_id: req.userId });
    
    if (!schedule) {
      schedule = new Schedule({
        user_id: req.userId,
        reminders: [reminderData]
      });
    } else {
      schedule.reminders.push(reminderData);
      schedule.updated_at = Date.now();
    }
    
    await schedule.save();
    res.status(201).json(schedule);
  } catch (error) {
    console.error('Error adding reminder:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Update reminder status
router.patch('/:reminderId', auth, async (req, res) => {
  try {
    const { reminderId } = req.params;
    const { status } = req.body;
    
    const schedule = await Schedule.findOne({ user_id: req.userId });
    if (!schedule) {
      return res.status(404).json({ message: 'Schedule not found' });
    }
    
    const reminder = schedule.reminders.id(reminderId);
    if (!reminder) {
      return res.status(404).json({ message: 'Reminder not found' });
    }
    
    reminder.status = status;
    if (status === 'completed') {
      reminder.last_taken = Date.now();
    }
    
    schedule.updated_at = Date.now();
    await schedule.save();
    
    res.json(schedule);
  } catch (error) {
    console.error('Error updating reminder:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Delete a reminder
router.delete('/:reminderId', auth, async (req, res) => {
  try {
    const { reminderId } = req.params;
    
    const schedule = await Schedule.findOne({ user_id: req.userId });
    if (!schedule) {
      return res.status(404).json({ message: 'Schedule not found' });
    }
    
    schedule.reminders = schedule.reminders.filter(
      reminder => reminder._id.toString() !== reminderId
    );
    
    schedule.updated_at = Date.now();
    await schedule.save();
    
    res.json({ message: 'Reminder deleted successfully' });
  } catch (error) {
    console.error('Error deleting reminder:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router;
