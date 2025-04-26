from QA import ConversationalQA
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Singleton instance
_qa_instance = None

def get_qa_instance():
    """Get or create the singleton instance of ConversationalQA"""
    global _qa_instance
    if _qa_instance is None:
        logger.info("Creating ConversationalQA instance")
        _qa_instance = ConversationalQA()
        
        # Initialize with a default news-focused passage
        default_passage = """
            Recent global news trends include advancements in technology, climate change developments, economic shifts, and international relations. 
            Technology companies are focusing on AI developments, cloud computing, and cybersecurity. 
            Climate change continues to be a major focus with countries working toward carbon reduction goals.
            Global economies are navigating inflation concerns, supply chain challenges, and digital transformation.
            International relations remain complex with ongoing diplomatic efforts in various regions.
            The media landscape continues to evolve with digital platforms playing an increasingly important role in news distribution.
            Health news has been centered around public health infrastructure, medical research, and healthcare access.
        """
        _qa_instance.set_passage(default_passage)
        logger.info("Initialized QA system with default news passage")
    return _qa_instance 