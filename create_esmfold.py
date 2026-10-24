import torch
import esm
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple, Optional, Union
import logging
import os
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ESMProteinEmbedder:
    """ESM protein embedding generator using local .pt files"""
    
    def __init__(self, 
                 model_path: str = ".cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt",
                 device: Optional[str] = None):
        """
        Initialize ESM embedder
        
        Parameters:
            model_path: Path to local .pt model file
            device: Specify device ("cuda", "cpu", "cuda:0")
        """
        self.model_path = model_path
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from local file: {model_path}")
        start_time = time.time()
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model from local file
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
        
        # Move to specified device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get batch converter and padding index
        self.batch_converter = self.alphabet.get_batch_converter()
        self.padding_idx = self.alphabet.padding_idx
        
        # Record model information
        self.model_name = Path(model_path).stem
        self.num_layers = self.model.num_layers
        self.embed_dim = self.model.embed_dim
        
        logger.info(f"Model loaded successfully, time: {time.time() - start_time:.2f} seconds")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Number of layers: {self.num_layers}")
        logger.info(f"Embedding dimension: {self.embed_dim}")
        logger.info(f"Vocabulary size: {len(self.alphabet)}")
        
        # Print CUDA memory info (if using GPU)
        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def read_sequences_triplet_format(self, input_path: str) -> List[Tuple[str, str, str]]:
        """
        Read protein file in triplet format
        
        Format:
            Line 1: Identifier (starts with '>')
            Line 2: Protein sequence
            Line 3: Comment (optional, may or may not start with '#')
            
        Example:
            >P12345
            MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
            This is a comment line
            
        Parameters:
            input_path: Path to input file
            
        Returns:
            List of tuples containing (identifier, sequence, comment)
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        logger.info(f"Reading triplet format file: {input_path}")
        
        sequences = []
        line_count = 0
        
        with open(input_path, 'r') as f:
            lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip empty lines
                if not line:
                    i += 1
                    continue
                
                # Find identifier line (starts with '>')
                if line.startswith('>'):
                    identifier = line[1:].strip()  # Remove '>'
                    line_count += 1
                    
                    # Read sequence line
                    if i + 1 < len(lines):
                        sequence = lines[i + 1].strip().upper()
                        line_count += 1
                        
                        # Read comment line (may or may not start with '#')
                        if i + 2 < len(lines):
                            comment = lines[i + 2].strip()
                            # Remove '#' if present at the beginning
                            if comment.startswith('#'):
                                comment = comment[1:].strip()
                            line_count += 1
                            i += 3  # Move to next group
                        else:
                            comment = ""
                            i += 2  # Only identifier and sequence
                        
                        # Validate sequence contains only standard amino acids
                        valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
                        sequence = ''.join([aa for aa in sequence if aa in valid_amino_acids])
                        
                        if len(sequence) >= 5:  # At least 5 amino acids
                            sequences.append((identifier, sequence, comment))
                        else:
                            logger.warning(f"Sequence '{identifier}' is too short or invalid, skipping")
                    else:
                        logger.warning(f"Identifier '{identifier}' has no corresponding sequence line")
                        i += 1
                else:
                    # Not an identifier line, skip
                    i += 1
        
        logger.info(f"Read {len(sequences)} protein sequences")
        logger.info(f"Processed {line_count} lines")
        
        # Print first few sequences as examples
        for i, (identifier, sequence, comment) in enumerate(sequences[:3]):
            logger.info(f"Example {i+1}: ID={identifier}, length={len(sequence)}, comment={comment[:50]}...")
        
        return sequences
    
    def read_sequences_fasta_format(self, input_path: str) -> List[Tuple[str, str, str]]:
        """
        Read protein file in FASTA format
        
        Format:
            >identifier optional comment
            Sequence line 1
            Sequence line 2
            ...
            
        Example:
            >P12345 This is a comment
            MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
            
        Parameters:
            input_path: Path to input file
            
        Returns:
            List of tuples containing (identifier, sequence, comment)
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        logger.info(f"Reading FASTA format file: {input_path}")
        
        sequences = []
        current_id = ""
        current_comment = ""
        current_seq = []
        
        with open(input_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # If line starts with '>', it's a header line
                if line.startswith('>'):
                    # Save previous sequence if exists
                    if current_id and current_seq:
                        sequence = ''.join(current_seq).upper()
                        
                        # Validate sequence contains only standard amino acids
                        valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
                        sequence = ''.join([aa for aa in sequence if aa in valid_amino_acids])
                        
                        if len(sequence) >= 5:  # At least 5 amino acids
                            sequences.append((current_id, sequence, current_comment))
                        else:
                            logger.warning(f"Sequence '{current_id}' is too short or invalid, skipping")
                    
                    # Start new sequence
                    header = line[1:].strip()
                    
                    # Split header into ID and optional comment
                    parts = header.split(maxsplit=1)
                    if len(parts) >= 1:
                        current_id = parts[0]
                        current_comment = parts[1] if len(parts) > 1 else ""
                    else:
                        current_id = f"seq_{line_num}"
                        current_comment = ""
                    
                    current_seq = []
                else:
                    # Sequence line
                    current_seq.append(line)
        
        # Save the last sequence
        if current_id and current_seq:
            sequence = ''.join(current_seq).upper()
            
            # Validate sequence contains only standard amino acids
            valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
            sequence = ''.join([aa for aa in sequence if aa in valid_amino_acids])
            
            if len(sequence) >= 5:  # At least 5 amino acids
                sequences.append((current_id, sequence, current_comment))
            else:
                logger.warning(f"Sequence '{current_id}' is too short or invalid, skipping")
        
        logger.info(f"Read {len(sequences)} protein sequences")
        
        # Print first few sequences as examples
        for i, (identifier, sequence, comment) in enumerate(sequences[:3]):
            logger.info(f"Example {i+1}: ID={identifier}, length={len(sequence)}, comment={comment[:50]}...")
        
        return sequences
    
    def read_sequences_tab_format(self, input_path: str, delimiter: str = '\t') -> List[Tuple[str, str, str]]:
        """
        Read protein file in tabular format
        
        Format:
            ID\tSequence\tComment
            ID\tSequence\tComment
            ...
            
        Parameters:
            input_path: Path to input file
            delimiter: Column delimiter (default: tab)
            
        Returns:
            List of tuples containing (identifier, sequence, comment)
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        logger.info(f"Reading tabular format file: {input_path}")
        
        sequences = []
        
        with open(input_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comment lines
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(delimiter)
                
                if len(parts) >= 2:
                    identifier = parts[0].strip()
                    sequence = parts[1].strip().upper()
                    comment = parts[2].strip() if len(parts) > 2 else ""
                    
                    # Validate sequence contains only standard amino acids
                    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
                    sequence = ''.join([aa for aa in sequence if aa in valid_amino_acids])
                    
                    if len(sequence) >= 5:  # At least 5 amino acids
                        sequences.append((identifier, sequence, comment))
                    else:
                        logger.warning(f"Sequence '{identifier}' is too short or invalid, skipping")
                else:
                    logger.warning(f"Line {line_num}: Invalid format, skipping")
        
        logger.info(f"Read {len(sequences)} protein sequences")
        
        # Print first few sequences as examples
        for i, (identifier, sequence, comment) in enumerate(sequences[:3]):
            logger.info(f"Example {i+1}: ID={identifier}, length={len(sequence)}, comment={comment[:50]}...")
        
        return sequences
    
    def read_sequences(self, input_path: str, format: str = "triplet") -> List[Tuple[str, str, str]]:
        """
        Read protein sequences from file based on specified format
        
        Parameters:
            input_path: Path to input file
            format: Input file format ("triplet", "fasta", or "tabular")
            
        Returns:
            List of tuples containing (identifier, sequence, comment)
        """
        format = format.lower()
        
        if format == "triplet":
            return self.read_sequences_triplet_format(input_path)
        elif format == "fasta":
            return self.read_sequences_fasta_format(input_path)
        elif format == "tabular":
            return self.read_sequences_tab_format(input_path)
        else:
            raise ValueError(f"Unsupported format: {format}. Supported formats: triplet, fasta, tabular")
    
    def prepare_batches(self, sequences: List[Tuple[str, str, str]], 
                        batch_size: int = 8) -> List[List[Tuple[str, str]]]:
        """
        Prepare batch data (ignore comments)
        
        Parameters:
            sequences: List of (identifier, sequence, comment) tuples
            batch_size: Size of each batch
            
        Returns:
            List of batches, each batch contains (identifier, sequence) tuples
        """
        # Take only identifier and sequence
        seq_data = [(identifier, sequence) for identifier, sequence, _ in sequences]
        
        # Optional: sort by sequence length for efficiency
        seq_data.sort(key=lambda x: len(x[1]), reverse=True)
        
        # Create batches
        batches = []
        for i in range(0, len(seq_data), batch_size):
            batch = seq_data[i:i + batch_size]
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} batches")
        return batches
    
    def generate_embeddings(self, 
                           sequences: List[Tuple[str, str, str]],
                           batch_size: int = 4,
                           layers: List[int] = None,
                           pooling: str = "mean",
                           include_per_residue: bool = False,
                           include_attention: bool = False) -> Dict:
        """
        Generate embeddings for a list of protein sequences
        
        Parameters:
            sequences: List of (identifier, sequence, comment) tuples
            batch_size: Batch size (adjust based on GPU memory)
            layers: List of layers to extract, None for the last layer
            pooling: Pooling method ("mean", "sum", "max", "cls", "last")
            include_per_residue: Whether to include per-residue embeddings
            include_attention: Whether to include attention weights
            
        Returns:
            Dictionary containing embeddings
        """
        if not sequences:
            logger.warning("Sequence list is empty")
            return {}
        
        if layers is None:
            layers = [self.num_layers]  # Default to the last layer
        
        logger.info(f"Starting to generate embeddings for {len(sequences)} proteins")
        logger.info(f"Batch size: {batch_size}, Layers: {layers}, Pooling: {pooling}")
        
        # Initialize results dictionary
        results = {
            'identifiers': [],
            'sequences': [],
            'comments': [],
            'sequence_lengths': [],
            'pooled_embeddings': {},  # Pooled embeddings
            'per_residue_embeddings': {} if include_per_residue else None,
            'attention_weights': {} if include_attention else None,
            'metadata': {
                'model': self.model_name,
                'layers': layers,
                'pooling': pooling,
                'embedding_dim': self.embed_dim,
                'device': str(self.device),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Initialize pooled embeddings list for each layer
        for layer in layers:
            results['pooled_embeddings'][f'layer_{layer}'] = []
        
        # Prepare batches
        batches = self.prepare_batches(sequences, batch_size)
        
        start_time = time.time()
        total_sequences = 0
        
        # Process each batch
        for batch_idx, batch in enumerate(batches):
            batch_identifiers = [identifier for identifier, _ in batch]
            batch_sequences = [sequence for _, sequence in batch]
            
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} "
                       f"(contains {len(batch)} sequences)")
            
            # Convert batch to model input
            batch_labels, batch_strs, batch_tokens = self.batch_converter(batch)
            batch_tokens = batch_tokens.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                if include_attention:
                    # If attention weights are needed
                    outputs = self.model(
                        batch_tokens, 
                        repr_layers=layers,
                        return_contacts=include_attention
                    )
                    
                    if include_attention and 'contacts' in outputs:
                        attention_weights = outputs['contacts'].cpu().numpy()
                else:
                    # Only need embeddings
                    outputs = self.model(batch_tokens, repr_layers=layers)
            
            # Process output for each sequence
            for seq_idx in range(len(batch)):
                seq_id = batch_labels[seq_idx]
                sequence = batch_sequences[seq_idx]
                
                # Find original comment
                comment = ""
                for identifier, seq, comm in sequences:
                    if identifier == seq_id:
                        comment = comm
                        break
                
                # Store basic information
                results['identifiers'].append(seq_id)
                results['sequences'].append(sequence)
                results['comments'].append(comment)
                results['sequence_lengths'].append(len(sequence))
                
                # Process each layer
                for layer in layers:
                    layer_key = f'layer_{layer}'
                    
                    # Get representation for this layer
                    token_repr = outputs["representations"][layer][seq_idx]
                    
                    # Calculate valid tokens (remove special tokens and padding)
                    mask = batch_tokens[seq_idx] != self.padding_idx
                    seq_len = mask.sum().item()
                    
                    if seq_len < 2:  # Only special tokens
                        logger.warning(f"Sequence '{seq_id}' is too short, using CLS token")
                        valid_repr = token_repr[0]  # Use CLS token
                        pooled_emb = valid_repr.cpu().numpy()
                    else:
                        # Extract valid tokens (remove <cls> and <eos>)
                        valid_repr = token_repr[1:seq_len-1]
                        
                        # Apply pooling method
                        if pooling == "mean":
                            pooled_emb = valid_repr.mean(dim=0).cpu().numpy()
                        elif pooling == "sum":
                            pooled_emb = valid_repr.sum(dim=0).cpu().numpy()
                        elif pooling == "max":
                            pooled_emb, _ = valid_repr.max(dim=0)
                            pooled_emb = pooled_emb.cpu().numpy()
                        elif pooling == "cls":
                            pooled_emb = token_repr[0].cpu().numpy()
                        elif pooling == "last":
                            pooled_emb = token_repr[seq_len-2].cpu().numpy()  # Token before <eos>
                        else:
                            raise ValueError(f"Unsupported pooling method: {pooling}")
                        
                        # Save per-residue embeddings (if needed)
                        if include_per_residue and layer == layers[-1]:  # Only save for the last layer
                            if 'per_residue_embeddings' not in results:
                                results['per_residue_embeddings'] = {}
                            results['per_residue_embeddings'][seq_id] = valid_repr.cpu().numpy()
                    
                    # Store pooled embedding
                    results['pooled_embeddings'][layer_key].append(pooled_emb)
                
                # Save attention weights (if needed)
                if include_attention and 'attention_weights' in outputs:
                    if seq_idx < len(attention_weights):
                        if 'attention_weights' not in results:
                            results['attention_weights'] = {}
                        results['attention_weights'][seq_id] = attention_weights[seq_idx]
                
                total_sequences += 1
            
            # Free GPU memory (if using GPU)
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Convert all lists to numpy arrays
        for layer_key in results['pooled_embeddings']:
            results['pooled_embeddings'][layer_key] = np.array(results['pooled_embeddings'][layer_key])
        
        logger.info(f"Embedding generation completed, time: {time.time() - start_time:.2f} seconds")
        logger.info(f"Processed {total_sequences} protein sequences")
        
        # Print statistics
        if results['pooled_embeddings']:
            first_layer = list(results['pooled_embeddings'].keys())[0]
            sample_emb = results['pooled_embeddings'][first_layer][0]
            logger.info(f"Embedding dimension: {sample_emb.shape[0]}")
        
        return results
    
    def save_embeddings(self, 
                       embeddings: Dict, 
                       output_dir: str, 
                       prefix: str = "protein_embeddings") -> Dict[str, str]:
        """
        Save embedding results to multiple files
        
        Parameters:
            embeddings: Embedding dictionary
            output_dir: Output directory
            prefix: Filename prefix
            
        Returns:
            Dictionary of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # 1. Save main embedding data (NPZ format)
        npz_path = output_dir / f"{prefix}.npz"
        
        # Prepare data to save
        save_data = {
            'identifiers': np.array(embeddings['identifiers'], dtype=object),
            'sequences': np.array(embeddings['sequences'], dtype=object),
            'sequence_lengths': np.array(embeddings['sequence_lengths']),
        }
        
        # Add pooled embeddings
        for layer_key, layer_data in embeddings['pooled_embeddings'].items():
            save_data[layer_key] = layer_data
        
        # Add per-residue embeddings (if exist)
        if embeddings.get('per_residue_embeddings'):
            # Note: Per-residue embeddings have different lengths per sequence,
            # so we save them in a separate file as a dictionary
            per_residue_path = output_dir / f"{prefix}_per_residue.npz"
            np.savez_compressed(per_residue_path, **embeddings['per_residue_embeddings'])
            saved_files['per_residue'] = str(per_residue_path)
            logger.info(f"Per-residue embeddings saved to: {per_residue_path}")
        
        # Add comments (if exist)
        if 'comments' in embeddings and embeddings['comments']:
            save_data['comments'] = np.array(embeddings['comments'], dtype=object)
        
        # Save main data
        np.savez_compressed(npz_path, **save_data)
        saved_files['main'] = str(npz_path)
        logger.info(f"Main embedding data saved to: {npz_path}")
        
        # 2. Save CSV format (for easy viewing)
        if embeddings['pooled_embeddings']:
            last_layer_key = list(embeddings['pooled_embeddings'].keys())[-1]
            pooled_emb = embeddings['pooled_embeddings'][last_layer_key]
            
            # Create DataFrame
            df_data = {
                'identifier': embeddings['identifiers'],
                'sequence': embeddings['sequences'],
                'length': embeddings['sequence_lengths'],
            }
            
            # Add embedding dimensions as columns
            for i in range(pooled_emb.shape[1]):
                df_data[f'embedding_{i:04d}'] = pooled_emb[:, i]
            
            # Add comments (if exist)
            if 'comments' in embeddings and embeddings['comments']:
                df_data['comment'] = embeddings['comments']
            
            df = pd.DataFrame(df_data)
            csv_path = output_dir / f"{prefix}_pooled.csv"
            df.to_csv(csv_path, index=False)
            saved_files['csv'] = str(csv_path)
            logger.info(f"Pooled embeddings CSV saved to: {csv_path}")
        
        # 3. Save metadata
        meta_path = output_dir / f"{prefix}_metadata.json"
        metadata = embeddings.get('metadata', {})
        metadata['num_sequences'] = len(embeddings['identifiers'])
        metadata['saved_files'] = list(saved_files.keys())
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        saved_files['metadata'] = str(meta_path)
        logger.info(f"Metadata saved to: {meta_path}")
        
        # 4. Save statistics
        stats_path = output_dir / f"{prefix}_stats.txt"
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"Protein Embedding Generation Statistics\n")
            f.write(f"========================================\n")
            f.write(f"Generation time: {metadata.get('timestamp', 'N/A')}\n")
            f.write(f"Model: {metadata.get('model', 'N/A')}\n")
            f.write(f"Device: {metadata.get('device', 'N/A')}\n")
            f.write(f"Number of sequences: {len(embeddings['identifiers'])}\n")
            f.write(f"Embedding dimension: {metadata.get('embedding_dim', 'N/A')}\n")
            f.write(f"Pooling method: {metadata.get('pooling', 'N/A')}\n")
            f.write(f"Layers: {metadata.get('layers', [])}\n")
            
            if 'sequence_lengths' in embeddings and embeddings['sequence_lengths']:
                lengths = embeddings['sequence_lengths']
                f.write(f"\nSequence length statistics:\n")
                f.write(f"  Shortest: {min(lengths)} amino acids\n")
                f.write(f"  Longest: {max(lengths)} amino acids\n")
                f.write(f"  Average: {np.mean(lengths):.1f} amino acids\n")
                f.write(f"  Median: {np.median(lengths):.1f} amino acids\n")
            
            f.write(f"\nSaved files:\n")
            for file_type, file_path in saved_files.items():
                f.write(f"  {file_type}: {Path(file_path).name}\n")
        
        saved_files['stats'] = str(stats_path)
        logger.info(f"Statistics saved to: {stats_path}")
        
        return saved_files
    
    def generate_single_embedding(self, 
                                 identifier: str, 
                                 sequence: str, 
                                 layer: int = None,
                                 pooling: str = "mean") -> np.ndarray:
        """
        Generate embedding for a single protein sequence
        
        Parameters:
            identifier: Protein identifier
            sequence: Protein sequence
            layer: Layer to use, None for the last layer
            pooling: Pooling method
            
        Returns:
            Embedding vector
        """
        if layer is None:
            layer = self.num_layers
        
        # Prepare data
        data = [(identifier, sequence)]
        
        # Convert to model input
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(tokens, repr_layers=[layer])
        
        # Get representation
        token_repr = outputs["representations"][layer][0]
        
        # Calculate valid tokens
        mask = tokens[0] != self.padding_idx
        seq_len = mask.sum().item()
        
        if seq_len < 2:
            return token_repr[0].cpu().numpy()
        
        # Extract valid tokens and pool
        valid_repr = token_repr[1:seq_len-1]
        
        if pooling == "mean":
            return valid_repr.mean(dim=0).cpu().numpy()
        elif pooling == "sum":
            return valid_repr.sum(dim=0).cpu().numpy()
        elif pooling == "max":
            max_vals, _ = valid_repr.max(dim=0)
            return max_vals.cpu().numpy()
        elif pooling == "cls":
            return token_repr[0].cpu().numpy()
        elif pooling == "last":
            return token_repr[seq_len-2].cpu().numpy()
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Generate protein embeddings using local ESM model"
    )
    
    parser.add_argument(
        "input_file",
        help="Path to input file"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=["triplet", "fasta", "tabular"],
        default="triplet",
        help="Input file format (default: triplet)"
    )
    
    parser.add_argument(
        "-o", "--output_dir",
        default="./protein_embeddings",
        help="Output directory (default: ./protein_embeddings)"
    )
    
    parser.add_argument(
        "-m", "--model_path",
        default=".cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt",
        help="ESM model file path (default: .cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt)"
    )
    
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=4,
        help="Batch size (default: 4, adjust based on GPU memory)"
    )
    
    parser.add_argument(
        "-p", "--pooling",
        default="mean",
        choices=["mean", "sum", "max", "cls", "last"],
        help="Pooling method (default: mean)"
    )
    
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Layers to extract (e.g., 6 12 24 36)"
    )
    
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "cuda:0", "cuda:1"],
        help="Compute device"
    )
    
    parser.add_argument(
        "--per_residue",
        action="store_true",
        help="Also save per-residue embeddings"
    )
    
    parser.add_argument(
        "--prefix",
        default="protein_embeddings",
        help="Output file prefix (default: protein_embeddings)"
    )
    
    parser.add_argument(
        "--delimiter",
        default="\t",
        help="Delimiter for tabular format (default: tab)"
    )
    
    args = parser.parse_args()
    
    # Print parameters
    print("="*50)
    print("ESM Protein Embedding Generator")
    print("="*50)
    print(f"Input file: {args.input_file}")
    print(f"Input format: {args.format}")
    print(f"Model file: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Pooling method: {args.pooling}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print(f"Save per-residue embeddings: {args.per_residue}")
    print("="*50)
    
    # Initialize embedder
    try:
        embedder = ESMProteinEmbedder(
            model_path=args.model_path,
            device=args.device
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the model file path is correct")
        return
    except Exception as e:
        print(f"Error initializing embedder: {e}")
        return
    
    # Read sequences
    try:
        if args.format == "triplet":
            sequences = embedder.read_sequences_triplet_format(args.input_file)
        elif args.format == "fasta":
            sequences = embedder.read_sequences_fasta_format(args.input_file)
        elif args.format == "tabular":
            sequences = embedder.read_sequences_tab_format(args.input_file, args.delimiter)
        else:
            print(f"Error: Unsupported format: {args.format}")
            return
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error reading sequences: {e}")
        return
    
    if not sequences:
        print("Error: No valid protein sequences were read")
        return
    
    # Set layers to extract
    if args.layers:
        layers = args.layers
    else:
        # Default: extract multiple representative layers
        layers = [6, 12, 24, embedder.num_layers]
    
    # Generate embeddings
    print("Generating embeddings...")
    try:
        embeddings = embedder.generate_embeddings(
            sequences=sequences,
            batch_size=args.batch_size,
            layers=layers,
            pooling=args.pooling,
            include_per_residue=args.per_residue,
            include_attention=False
        )
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        print("Try reducing batch size if you encounter memory errors")
        return
    
    # Save embeddings
    print("Saving embedding results...")
    try:
        saved_files = embedder.save_embeddings(
            embeddings=embeddings,
            output_dir=args.output_dir,
            prefix=args.prefix
        )
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        return
    
    # Print completion information
    print("\n" + "="*50)
    print("Embedding generation completed successfully!")
    print("="*50)
    
    print(f"\nProcessed {len(embeddings['identifiers'])} proteins")
    
    # Display saved files
    print(f"\nSaved files:")
    for file_type, file_path in saved_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
            print(f"  {file_type}: {file_path} ({file_size:.2f} MB)")
        else:
            print(f"  {file_type}: {file_path} (file not found)")
    
    # Display example embedding
    if embeddings['pooled_embeddings']:
        last_layer_key = list(embeddings['pooled_embeddings'].keys())[-1]
        sample_embedding = embeddings['pooled_embeddings'][last_layer_key][0]
        print(f"\nSample embedding shape: {sample_embedding.shape}")
        print(f"Sample embedding (first 5 values): {sample_embedding[:5]}")




if __name__ == "__main__":
    # Create example files (uncomment to create)
    # create_example_files()
    
    # Run main CLI
    main()