from openai import OpenAI
import base64
import gradio as gr
from dotenv import load_dotenv
import os
from PIL import Image
import io
import logging
from pydantic import BaseModel, Field
from typing import List

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI()

# Simplified model without validation constraints
class HairAnalysis(BaseModel):
    hair_type: str = Field(..., description="The type of hair (straight, wavy, curly, coily)")
    hair_texture: str = Field(..., description="The texture of the hair (fine, medium, coarse)")
    scalp_condition: str = Field(..., description="The condition of the scalp")
    visible_issues: List[str] = Field(..., description="List of visible issues with the hair")
    health_score: int = Field(..., description="Hair health score (must be between 1 and 10)")
    recommendations: List[str] = Field(..., description="List of recommendations for hair care")

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def analyze_hair(image):
    if image is None:
        return "Please upload an image for analysis."
    
    try:
        img_base64 = encode_image(image)
        logger.debug("Successfully encoded image to base64")
        
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a professional hair analysis system. Analyze the image and provide structured information about the hair and scalp.
                    Important scoring guidelines:
                    - Health score must be an integer from 1 to 10
                    - Score meanings:
                      1-3: Poor condition
                      4-6: Average condition
                      7-8: Good condition
                      9-10: Excellent condition"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this hair/scalp image and provide a detailed hair analysis. Remember to provide a health score between 1 and 10 only."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            response_format=HairAnalysis
        )
        
        result = completion.choices[0].message.parsed
        
        formatted_result = f"""
Hair Analysis Results:

1. Hair Type: {result.hair_type}
2. Hair Texture: {result.hair_texture}
3. Scalp Condition: {result.scalp_condition}
4. Visible Issues:
{chr(10).join('   • ' + issue for issue in result.visible_issues) if result.visible_issues else '   No major issues detected'}
5. Hair Health Score: {result.health_score}/10
6. Recommendations:
{chr(10).join('   • ' + rec for rec in result.recommendations)}
"""
        
        return formatted_result
    
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return f"Error analyzing image: {str(e)}"

def create_interface():
    interface = gr.Interface(
        fn=analyze_hair,
        inputs=gr.Image(
            label="Upload Image or Use Camera",
            type="pil",
            sources=["upload", "webcam"]
        ),
        outputs=gr.Textbox(
            label="Analysis Result",
            lines=15
        ),
        title="AI Hair & Scalp Analyzer",
        description="Upload an image or use your camera to capture your hair/scalp for analysis."
    )
    
    return interface

if __name__ == "__main__":
    logger.info("Starting Hair Analysis Application")
    iface = create_interface()
    iface.launch(debug=True)