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
    generative AI model and includes a user interface for generating text. The following 
    sections will cover the project's background, the model selected, evaluation metrics, 
    results, and conclusions.
    """
    story.append(Paragraph(introduction, styles['BodyText']))
    story.append(Spacer(1, 12))
    
    # Model Details
    model_details = """
    <b>Model Details:</b><br/>
    The text generation application was built using the <b>Falcon/Gemini/Bloom</b> model. 
    These models are known for their ability to generate coherent and contextually relevant 
    text based on the input provided. The model was fine-tuned on a specific dataset to 
    optimize its performance for the task at hand.
    """
    story.append(Paragraph(model_details, styles['BodyText']))
    story.append(Spacer(1, 12))
    
    # Evaluation Metrics
    evaluation_metrics = """
    <b>Evaluation Metrics:</b><br/>
    The performance of the text generation application was evaluated using several key 
    metrics, including Perplexity, BLEU score, and ROUGE score. These metrics provide 
    insight into the model's ability to generate fluent and accurate text.
    """
    story.append(Paragraph(evaluation_metrics, styles['BodyText']))
    story.append(Spacer(1, 12))
    
    # Results
    results = """
    <b>Results:</b><br/>
    The evaluation results showed that the model performed well across all metrics, 
    with a low Perplexity score indicating fluent text generation, and high BLEU 
    and ROUGE scores demonstrating the accuracy and relevance of the generated text.
    """
    story.append(Paragraph(results, styles['BodyText']))
    story.append(Spacer(1, 12))
    
    # Conclusion
    conclusion = """
    <b>Conclusion:</b><br/>
    The text generation application successfully met the project's objectives, providing 
    a tool capable of generating high-quality text. Future improvements could include 
    expanding the dataset and exploring more advanced models to further enhance performance.
    """
    story.append(Paragraph(conclusion, styles['BodyText']))
    story.append(Spacer(1, 12))
    
    # Build the PDF
    pdf.build(story)

if __name__ == "__main__":
    create_pdf_report("report.pdf")
