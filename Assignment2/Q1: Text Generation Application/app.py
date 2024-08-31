from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load the model and tokenizer
model_name = "bigscience/bloom"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/', methods=['GET', 'POST'])
def generate_text():
    if request.method == 'POST':
        # Get the prompt from the user
        prompt = request.form['prompt']
        
        # Tokenize and generate text
        inputs = tokenizer(prompt, return_tensors='pt')
        outputs = model.generate(inputs['input_ids'], max_length=150)
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Render the result on the webpage
        return render_template('index.html', prompt=prompt, generated_text=generated_text)
    
    # Render the form initially
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
