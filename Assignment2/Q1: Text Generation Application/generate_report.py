from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

def create_pdf_report(file_path):
    # Create the PDF document
    pdf = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = "Text Generation Application Report"
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 12))
    
    # Introduction
    introduction = """
    This report provides a comprehensive overview of the development and evaluation 
    of the text generation application. The application was developed using an open-source 
    generative AI model and includes a user interface for generating text based on prompts 
    related to Sustainable Development Goals (SDGs). The following sections will cover the 
    project's background, the model selected, evaluation metrics, results, and conclusions.
    """
    story.append(Paragraph(introduction, styles['BodyText']))
    story.append(Spacer(1, 12))
    
    # Model Details
    model_details = """
    <b>Model Details:</b><br/>
    The text generation application was built using the <b>Falcon/Gemini/Bloom</b> model. 
    These models are known for their ability to generate coherent and contextually relevant 
    text based on input prompts. The model was fine-tuned on a dataset consisting of prompts 
    related to various SDGs, aiming to produce outputs that are relevant and informative 
    in the context of global development challenges.
    """
    story.append(Paragraph(model_details, styles['BodyText']))
    story.append(Spacer(1, 12))
    
    # Dataset Details
    dataset_details = """
    <b>Dataset Details:</b><br/>
    The dataset used in this project consists of prompts related to Sustainable Development 
    Goals (SDGs). These prompts cover topics such as clean water and sanitation, affordable 
    and clean energy, quality education, gender equality, climate action, and more. The 
    prompts were specifically chosen to test the model's ability to generate relevant and 
    insightful text on global development issues.
    """
    story.append(Paragraph(dataset_details, styles['BodyText']))
    story.append(Spacer(1, 12))
    
    # Evaluation Metrics
    evaluation_metrics = """
    <b>Evaluation Metrics:</b><br/>
    The performance of the text generation application was evaluated using several key 
    metrics, including coherence, relevance, and grammatical correctness. These metrics 
    provide insight into the model's ability to generate fluent, contextually appropriate, 
    and informative text in response to the SDG-related prompts.
    """
    story.append(Paragraph(evaluation_metrics, styles['BodyText']))
    story.append(Spacer(1, 12))
    
    # Results
    results = """
    <b>Results:</b><br/>
    The evaluation results showed that the model performed well in generating text 
    that is coherent, contextually relevant, and grammatically correct. The model 
    effectively responded to a wide range of SDG-related prompts, demonstrating its 
    capability to support discussions and initiatives focused on sustainable development.
    """
    story.append(Paragraph(results, styles['BodyText']))
    story.append(Spacer(1, 12))
    
    # Conclusion
    conclusion = """
    <b>Conclusion:</b><br/>
    The text generation application successfully met the project's objectives, providing 
    a tool capable of generating high-quality text related to Sustainable Development Goals. 
    Future improvements could include expanding the dataset and exploring more advanced 
    models to further enhance the application's performance and utility in the context 
    of global development.
    """
    story.append(Paragraph(conclusion, styles['BodyText']))
    story.append(Spacer(1, 12))
    
    # Build the PDF
    pdf.build(story)

if __name__ == "__main__":
    create_pdf_report("report.pdf")
