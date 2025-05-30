from QA import ConversationalQA

_qaInstance = None

def get_qa_instance() :
    global _qaInstance
    if _qaInstance is  None:
        _qaInstance =  ConversationalQA()
        
        defaultPassage =  "can't provide an accurate passage.."
        _qaInstance.set_passage(defaultPassage)
    return _qaInstance 