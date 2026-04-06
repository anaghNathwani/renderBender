# LaughterGPT

## Overview
LaughterGPT is a sophisticated AI model designed to generate humor and provide entertainment through jokes, funny stories, and memes. This documentation will guide you through the installation process, usage examples, and an overview of its features and architecture.

## Installation Instructions
To install LaughterGPT, follow the steps below:

1. **Clone the Repository**  
   Open your terminal and run the following command to clone the repository:
   ```bash
   git clone https://github.com/anaghNathwani/laughterGPT.git
   ```

2. **Navigate to the Directory**  
   Change to the directory of the cloned repository:
   ```bash
   cd laughterGPT
   ```

3. **Install Dependencies**  
   Use pip to install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Examples
Once LaughterGPT is installed, you can use it in the following ways:

### Generating a Joke
To generate a joke, use the following command:
```python
from laughterGPT import LaughterGPT

model = LaughterGPT()
print(model.get_joke())
```

### Creating a Funny Story
To create a funny story, use:
```python
story_prompt = "Write a funny story about a cat."
print(model.create_story(story_prompt))
```

## Features
- **Joke Generation**: Create random jokes in seconds.
- **Story Creation**: Generate humorous stories based on prompts.
- **Meme Generation**: Create funny memes with user-defined templates.
- **Custom Training**: Enhance the model with your own data for personalized humor.

## Architecture Details
LaughterGPT is built on a transformer architecture, specifically utilizing the following components:
- **Transformer Encoder-Decoder**: For understanding and generating text.
- **Attention Mechanisms**: Helps in evaluating the relevance of words in a phrase.
- **Pre-trained Models**: Fine-tuned on a vast dataset of jokes and humorous content.

The architecture is designed to ensure quick response times and high-quality output.

## Contributing
If you want to contribute to LaughterGPT, please fork the repository and submit a pull request with your enhancements or fixes.

## Usage
To traint the model, use either of the following two commands for best results:
1. python ai_agent.py train wikitext2
2. python ai_agent.py train wiki <topic name>

To chat with the model, use the following command for beest results:
python ai_agent.py generate "<your message here>"

## License
LaughterGPT is licensed under the MIT License. See the `LICENSE` file for more details.
