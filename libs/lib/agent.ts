import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';
 
export async function getWeather() {
  const { text } = await generateText({
    model: openai('gpt-4.1'),
    prompt: 'What is the weather like today?',
  });
 
  console.log(text);
}