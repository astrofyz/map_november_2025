"""
Script to classify street name origins using Llama (via Ollama).

First, install Ollama:
1. Visit https://ollama.ai and download Ollama for your OS
2. Or install via: curl -fsSL https://ollama.ai/install.sh | sh

Then download a model:
    ollama pull llama2
    # or for French, use:
    ollama pull mistral
    # or using Gemma 3:
    ollama pull gemma3:latest

Usage:
    python classify_with_llama.py                    # Process all entries
    python classify_with_llama.py --test            # Test on 10 random entries
    python classify_with_llama.py --test --test-n 5 # Test on 5 random entries
"""

import pandas as pd
import requests
import json
import time
from typing import Dict, Optional
import sys
import random

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:latest"  # Using Gemma 3 model
BATCH_SIZE = 10  # Process in batches to avoid overwhelming the API
DELAY_BETWEEN_REQUESTS = 0.5  # seconds


def classify_origin(orig_text: str, model: str = MODEL_NAME) -> Dict[str, str]:
    """
    Classify a street origin text using Llama via Ollama.
    
    Returns:
        {
            'is_person': 'yes' or 'no',
            'gender': 'M' (male), 'F' (female), 'N/A' (not a person), or 'U' (unknown)
        }
    """
    # Clean the text
    orig_text = str(orig_text).strip()
    if not orig_text or orig_text == 'nan':
        return {'is_person': 'no', 'gender': 'N/A'}
    
    # Create the prompt in French (since the data is in French)
    prompt = f"""Analyse le texte suivant qui décrit l'origine du nom d'une rue à Paris.

Texte: "{orig_text}"

Réponds UNIQUEMENT avec un JSON valide au format suivant:
{{
    "is_person": "yes" ou "no",
    "gender": "M" (homme), "F" (femme), "N/A" (si ce n'est pas une personne), ou "U" (incertain)
}}

Règles:
- Si le texte mentionne clairement une personne (nom, dates de naissance/mort, profession, titre), réponds "yes" pour is_person
- Si c'est un nom de propriétaire mais sans autres détails, c'est une personne
- Si c'est lié à un lieu, événement historique, quartier, voisinage, bâtiment, etc., réponds "no"
- Pour le genre, utilise "M" pour homme, "F" pour femme
- Si tu n'es pas sûr du genre, utilise "U"
- Si ce n'est pas une personne, utilise "N/A" pour le genre

Réponds SEULEMENT avec le JSON, rien d'autre:"""

    try:
        # Make request to Ollama
        response = requests.post(
            OLLAMA_BASE_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Lower temperature for more consistent results
                }
            },
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"Error: HTTP {response.status_code}")
            print(response.text)
            return {'is_person': 'error', 'gender': 'error'}
        
        result = response.json()
        answer = result.get('response', '').strip()
        
        # Try to extract JSON from the response
        # Sometimes the model adds extra text, so we look for JSON
        try:
            # Find JSON object in the response
            start_idx = answer.find('{')
            end_idx = answer.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = answer[start_idx:end_idx]
                classification = json.loads(json_str)
                
                # Normalize values
                is_person = str(classification.get('is_person', 'no')).lower()
                gender = str(classification.get('gender', 'N/A')).upper()
                
                # Validate
                if is_person not in ['yes', 'no']:
                    is_person = 'no'
                if gender not in ['M', 'F', 'N/A', 'U']:
                    gender = 'N/A' if is_person == 'no' else 'U'
                
                return {
                    'is_person': 'yes' if is_person == 'yes' else 'no',
                    'gender': gender
                }
            else:
                # Fallback: try to parse the whole response
                classification = json.loads(answer)
                return {
                    'is_person': 'yes' if str(classification.get('is_person', 'no')).lower() == 'yes' else 'no',
                    'gender': str(classification.get('gender', 'N/A')).upper()
                }
        except json.JSONDecodeError:
            # If JSON parsing fails, try to infer from text
            answer_lower = answer.lower()
            is_person = 'yes' if ('yes' in answer_lower or '"is_person": "yes"' in answer_lower) else 'no'
            gender = 'U'
            if 'gender": "m' in answer_lower or 'homme' in answer_lower:
                gender = 'M'
            elif 'gender": "f' in answer_lower or 'femme' in answer_lower:
                gender = 'F'
            elif is_person == 'no':
                gender = 'N/A'
            
            return {'is_person': is_person, 'gender': gender}
            
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return {'is_person': 'error', 'gender': 'error'}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {'is_person': 'error', 'gender': 'error'}


def test_classification(input_file: str = "streets_orig.csv",
                       model: str = MODEL_NAME,
                       n_samples: int = 10,
                       random_seed: Optional[int] = None):
    """
    Test classification on a random subset of entries.
    
    Args:
        input_file: Input CSV file path
        model: Ollama model name to use
        n_samples: Number of random samples to test
        random_seed: Random seed for reproducibility
    """
    print(f"🧪 Testing classification on {n_samples} random entries")
    print(f"Loading {input_file}...")
    
    df = pd.read_csv(input_file)
    
    if 'orig' not in df.columns:
        print("Error: 'orig' column not found in CSV")
        return
    
    # Filter out rows with missing 'orig' values
    df_valid = df[df['orig'].notna()].copy()
    
    if len(df_valid) == 0:
        print("Error: No valid entries found in 'orig' column")
        return
    
    if len(df_valid) < n_samples:
        print(f"Warning: Only {len(df_valid)} valid entries available, using all of them")
        n_samples = len(df_valid)
    
    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
    
    # Randomly sample entries
    sampled_indices = random.sample(range(len(df_valid)), n_samples)
    sampled_df = df_valid.iloc[sampled_indices].copy()
    
    # Get original indices from the full dataframe
    original_indices = df_valid.index[sampled_indices].tolist()
    
    print(f"Selected {n_samples} random entries (original indices: {original_indices})")
    print(f"Using model: {model}")
    print("=" * 80)
    print()
    
    results = []
    
    try:
        for i, (orig_idx, row) in enumerate(sampled_df.iterrows(), 1):
            orig_text = row['orig']
            print(f"[{i}/{n_samples}] Entry #{orig_idx}")
            print(f"  Text: {orig_text}")
            print(f"  Classifying...")
            
            result = classify_origin(orig_text, model)
            
            results.append({
                'index': orig_idx,
                'orig': orig_text,
                'is_person': result['is_person'],
                'gender': result['gender']
            })
            
            print(f"  ✅ Result: is_person={result['is_person']}, gender={result['gender']}")
            print()
            
            # Delay to avoid overwhelming the API
            time.sleep(DELAY_BETWEEN_REQUESTS)
        
        # Print summary table
        print("=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'#':<4} {'Is Person':<12} {'Gender':<8} {'Text Preview':<50}")
        print("-" * 80)
        
        person_count = 0
        male_count = 0
        female_count = 0
        
        for i, result in enumerate(results, 1):
            is_person = result['is_person']
            gender = result['gender']
            text_preview = result['orig'][:47] + "..." if len(result['orig']) > 50 else result['orig']
            
            print(f"{i:<4} {is_person:<12} {gender:<8} {text_preview}")
            
            if is_person == 'yes':
                person_count += 1
                if gender == 'M':
                    male_count += 1
                elif gender == 'F':
                    female_count += 1
        
        print("-" * 80)
        print(f"\nStatistics:")
        print(f"  Total tested: {n_samples}")
        print(f"  Named after person: {person_count} ({person_count/n_samples*100:.1f}%)")
        print(f"  Not named after person: {n_samples - person_count} ({(n_samples-person_count)/n_samples*100:.1f}%)")
        print(f"  Male: {male_count}")
        print(f"  Female: {female_count}")
        print(f"  Unknown/Other: {person_count - male_count - female_count}")
        print()
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted by user.")
        print(f"Results so far:")
        for result in results:
            print(f"  Entry #{result['index']}: is_person={result['is_person']}, gender={result['gender']}")
        return results
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        return results


def process_csv(input_file: str = "streets_orig.csv", 
                output_file: str = "streets_classified.csv",
                model: str = MODEL_NAME,
                start_index: int = 0,
                max_rows: Optional[int] = None):
    """
    Process the CSV file and classify all origins.
    
    Args:
        input_file: Input CSV file path
        output_file: Output CSV file path
        model: Ollama model name to use
        start_index: Start from this row (for resuming)
        max_rows: Maximum number of rows to process (None = all)
    """
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    if 'orig' not in df.columns:
        print("Error: 'orig' column not found in CSV")
        return
    
    # Initialize classification columns if they don't exist
    if 'is_person_llama' not in df.columns:
        df['is_person_llama'] = None
    if 'gender_llama' not in df.columns:
        df['gender_llama'] = None
    
    total_rows = len(df)
    end_index = total_rows if max_rows is None else min(start_index + max_rows, total_rows)
    
    print(f"Processing rows {start_index} to {end_index-1} of {total_rows}...")
    print(f"Using model: {model}")
    print(f"Press Ctrl+C to stop (progress will be saved)")
    print("-" * 60)
    
    try:
        for idx in range(start_index, end_index):
            if pd.notna(df.loc[idx, 'is_person_llama']):
                # Skip already classified rows
                print(f"Row {idx+1}/{total_rows}: Already classified, skipping...")
                continue
            
            orig_text = df.loc[idx, 'orig']
            print(f"Row {idx+1}/{total_rows}: Classifying...")
            
            result = classify_origin(orig_text, model)
            
            df.loc[idx, 'is_person_llama'] = result['is_person']
            df.loc[idx, 'gender_llama'] = result['gender']
            
            print(f"  Result: is_person={result['is_person']}, gender={result['gender']}")
            print(f"  Text: {orig_text[:80]}...")
            
            # Save progress periodically (every 10 rows)
            if (idx + 1) % 10 == 0:
                df.to_csv(output_file, index=False)
                print(f"  [Progress saved to {output_file}]")
            
            # Delay to avoid overwhelming the API
            time.sleep(DELAY_BETWEEN_REQUESTS)
        
        # Final save
        df.to_csv(output_file, index=False)
        print(f"\n✅ Classification complete! Results saved to {output_file}")
        
        # Print summary
        print("\nSummary:")
        print(f"Total processed: {end_index - start_index} rows")
        if 'is_person_llama' in df.columns:
            person_count = (df['is_person_llama'] == 'yes').sum()
            print(f"Named after person: {person_count}")
            print(f"Not named after person: {(df['is_person_llama'] == 'no').sum()}")
            if 'gender_llama' in df.columns:
                male_count = (df['gender_llama'] == 'M').sum()
                female_count = (df['gender_llama'] == 'F').sum()
                print(f"  - Male: {male_count}")
                print(f"  - Female: {female_count}")
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user. Saving progress...")
        df.to_csv(output_file, index=False)
        print(f"Progress saved to {output_file}")
        print(f"Resume by running with start_index={idx+1}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Classify street name origins using Llama')
    parser.add_argument('--input', default='streets_orig.csv', help='Input CSV file')
    parser.add_argument('--output', default='streets_classified.csv', help='Output CSV file')
    parser.add_argument('--model', default=MODEL_NAME, help='Ollama model name')
    parser.add_argument('--start', type=int, default=0, help='Start from this row index')
    parser.add_argument('--max', type=int, default=None, help='Maximum rows to process')
    parser.add_argument('--test', action='store_true', help='Run test on random subset (default: 10 entries)')
    parser.add_argument('--test-n', type=int, default=10, help='Number of samples for test mode (default: 10)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for test mode')
    
    args = parser.parse_args()
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            if args.model not in model_names:
                print(f"⚠️  Warning: Model '{args.model}' not found in Ollama.")
                print(f"Available models: {model_names}")
                print(f"To download a model, run: ollama pull {args.model}")
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    sys.exit(1)
    except requests.exceptions.RequestException:
        print("❌ Error: Cannot connect to Ollama. Is it running?")
        print("\nTo start Ollama:")
        print("1. Install it from https://ollama.ai")
        print("2. Run: ollama serve")
        print("3. Or just run: ollama pull gemma3:latest")
        sys.exit(1)
    
    # Run test mode if requested
    if args.test:
        test_classification(args.input, args.model, args.test_n, args.seed)
    else:
        process_csv(args.input, args.output, args.model, args.start, args.max)

