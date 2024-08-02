def initialize_initialization(task, subtask=None):
    if task == "text_generation":
        from nyuntam_adapt.tasks.text_generation import CausalLLM

        return CausalLLM

    elif task == "text_classification":
        from nyuntam_adapt.tasks.text_classification import SequenceClassification

        return SequenceClassification

    elif task == "Seq2Seq_tasks":
        if subtask == "translation":
            from nyuntam_adapt.tasks.translation import Translation

            return Translation

        elif subtask == "summarization":
            from nyuntam_adapt.tasks.summarization import Seq2Seq

            return Seq2Seq

    elif task in ["question_answering", "Question Answering"]:
        from nyuntam_adapt.tasks.question_answering import QuestionAnswering

        return QuestionAnswering

    elif task == "image_classification":
        from nyuntam_adapt.tasks.image_classification import ImageClassification

        return ImageClassification
    
    elif task == "object_detection":
        from nyuntam_adapt.tasks.object_detection_mmdet import ObjDetectionMmdet

        return ObjDetectionMmdet

    elif task == "image_segmentation":
        from nyuntam_adapt.tasks.image_segmentation_mmseg import  ImgSegmentationMmseg

        return ImgSegmentationMmseg
