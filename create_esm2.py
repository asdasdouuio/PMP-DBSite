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
import sys
import argparse
import re

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
        # Save original path for logging
        self.original_model_path = model_path
        
        # Check if path exists, if not try to find correct path
        self.model_path = self._find_model_file(model_path)
        
        # Verify model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        logger.info(f"Loading model from: {self.model_path}")
        logger.info(f"File exists: {os.path.exists(self.model_path)}")
        if os.path.exists(self.model_path):
            logger.info(f"File size: {os.path.getsize(self.model_path) / (1024**3):.2f} GB")
        
        start_time = time.time()
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model from local file
        try:
            logger.info("Starting to load ESM model...")
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(f"Model path used: {self.model_path}")
            logger.error(f"File exists: {os.path.exists(self.model_path)}")
            logger.error(f"Original path provided: {model_path}")
            raise
        
        # Move to specified device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get batch converter and padding index
        self.batch_converter = self.alphabet.get_batch_converter()
        self.padding_idx = self.alphabet.padding_idx
        
        # Record model information
        self.model_name = Path(self.model_path).stem
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
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Clean filename by removing invalid characters
        """
        # Remove or replace invalid filename characters
        invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
        sanitized = re.sub(invalid_chars, '_', filename)
        
        # Limit filename length (avoid too long)
        if len(sanitized) > 200:
            sanitized = sanitized[:100] + "..." + sanitized[-50:]
        
        # Remove leading and trailing dots or spaces
        sanitized = sanitized.strip('. ')
        
        # If filename is empty, use default name
        if not sanitized:
            sanitized = "unnamed_protein"
        
        return sanitized
    
    def _find_model_file(self, model_path: str) -> str:
        """
        Find the correct path for model file
        """
        # If input path is absolute and exists, return directly
        if os.path.isabs(model_path) and os.path.exists(model_path):
            logger.info(f"Using absolute path: {model_path}")
            return os.path.abspath(model_path)
        
        # If relative path, try multiple possible locations
        current_dir = os.getcwd()  # Current working directory
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Script directory
        
        # List of possible paths to try
        possible_paths = [
            model_path,
            os.path.join(current_dir, model_path),
            os.path.join(script_dir, model_path),
            "/home/HNNU25-WCX/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt",
            os.path.expanduser("~/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt"),
            os.path.join(current_dir, ".cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt"),
            os.path.join(script_dir, ".cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt"),
        ]
        
        # Remove duplicate paths
        unique_paths = []
        for path in possible_paths:
            if path not in unique_paths:
                unique_paths.append(path)
        
        # Find existing file
        found_path = None
        for path in unique_paths:
            if os.path.exists(path):
                found_path = os.path.abspath(path)
                logger.info(f"Found model file at: {found_path}")
                break
        
        if found_path:
            return found_path
        else:
            logger.error(f"Model file not found. Tried the following paths:")
            for path in unique_paths:
                if path:
                    logger.error(f"  - {path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def read_sequences_triplet_format(self, input_path: str) -> List[Tuple[str, str, str]]:
        """
        Read protein file in triplet format
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
                                
                            # ==========================================
                            # 核心修改：将 TE46 的 '+' 替换为 '1', '-' 替换为 '0'
                            # ==========================================
                            comment = comment.replace('+', '1').replace('-', '0')
                                
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
        
        return sequences
    
    def read_sequences_fasta_format(self, input_path: str) -> List[Tuple[str, str, str]]:
        """
        Read protein file in FASTA format
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
        
        return sequences
    
    def read_sequences_tab_format(self, input_path: str, delimiter: str = '\t') -> List[Tuple[str, str, str]]:
        """
        Read protein file in tabular format
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
        
        return sequences
    
    def read_sequences(self, input_path: str, format: str = "triplet") -> List[Tuple[str, str, str]]:
        """
        Read protein sequences from file based on specified format
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
    
    def generate_embedding_for_single_sequence(self, 
                                              identifier: str, 
                                              sequence: str,
                                              layer: int = None,
                                              pooling: str = "none") -> np.ndarray:
        """
        Generate embedding for a single protein sequence
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
        
        # Extract valid tokens
        valid_repr = token_repr[1:seq_len-1]
        
        # Return based on pooling strategy
        if pooling == "none":
            return valid_repr.cpu().numpy() # This returns the [seq_len, embed_dim] matrix needed for graphs
        elif pooling == "mean":
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
    
    def generate_embeddings_one_by_one(self,
                                      sequences: List[Tuple[str, str, str]],
                                      output_dir: str,
                                      layers: List[int] = None,
                                      pooling: str = "none",
                                      include_per_residue: bool = False) -> Dict:
        """
        Generate embeddings for protein sequences one by one
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not sequences:
            logger.warning("Sequence list is empty")
            return {}
        
        if layers is None:
            layers = [self.num_layers]  # Default to the last layer
        
        logger.info(f"Starting to generate embeddings for {len(sequences)} proteins, one by one")
        logger.info(f"Layers: {layers}, Pooling: {pooling}")
        
        # Initialize statistics
        stats = {
            'identifiers': [],
            'sequences': [],
            'labels': [], # 保存真实的0/1标签
            'sequence_lengths': [],
            'successful_count': 0,
            'failed_count': 0,
            'failed_proteins': [],
            'start_time': time.time(),
            'metadata': {
                'model': self.model_name,
                'layers': layers,
                'pooling': pooling,
                'embedding_dim': self.embed_dim,
                'device': str(self.device),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Process sequences one by one
        for idx, (identifier, sequence, comment) in enumerate(sequences):
            
            try:
                # Generate embedding for each layer
                for layer in layers:
                    # Generate embedding
                    embedding = self.generate_embedding_for_single_sequence(
                        identifier=identifier,
                        sequence=sequence,
                        layer=layer,
                        pooling=pooling
                    )
                    
                    # Clean filename
                    safe_identifier = self._sanitize_filename(identifier)
                    
                    # Build filename
                    if len(layers) > 1:
                        npy_filename = f"{safe_identifier}_layer{layer}.npy"
                    else:
                        npy_filename = f"{safe_identifier}.npy"
                    
                    npy_path = output_dir / npy_filename
                    
                    # Save as .npy file
                    np.save(npy_path, embedding)
                    
                # Update statistics
                stats['identifiers'].append(identifier)
                stats['sequences'].append(sequence)
                stats['labels'].append(comment)
                stats['sequence_lengths'].append(len(sequence))
                stats['successful_count'] += 1
                
                # Log progress every 10 proteins
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(sequences)} proteins")
                
            except Exception as e:
                logger.error(f"Failed to generate embedding for protein '{identifier}': {e}")
                stats['failed_count'] += 1
                stats['failed_proteins'].append((identifier, str(e)))
            
            # If GPU, clear cache periodically
            if self.device.type == 'cuda' and (idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
        
        # Calculate total time
        stats['total_time'] = time.time() - stats['start_time']
        
        logger.info(f"Embedding generation completed")
        logger.info(f"Total time: {stats['total_time']:.2f} seconds")
        logger.info(f"Successful: {stats['successful_count']}, Failed: {stats['failed_count']}")
        
        return stats
    
    def save_metadata_and_stats(self,
                               stats: Dict,
                               output_dir: str,
                               prefix: str = "protein_embeddings") -> Dict[str, str]:
        """
        Save metadata and statistics
        """
        output_dir = Path(output_dir)
        
        saved_files = {}
        
        # 1. 自动导出一份 CSV，把 PDB_ID、序列和 0/1 标签绑在一起
        if stats.get('labels'):
            csv_path = output_dir / f"{prefix}_labels.csv"
            df = pd.DataFrame({
                'identifier': stats['identifiers'],
                'sequence': stats['sequences'],
                'label_01': stats['labels']
            })
            df.to_csv(csv_path, index=False)
            saved_files['labels_csv'] = str(csv_path)
            logger.info(f"0/1 Labels perfectly extracted and saved to: {csv_path}")

        # 2. Save metadata
        meta_path = output_dir / f"{prefix}_metadata.json"
        metadata = stats.get('metadata', {})
        metadata['num_sequences'] = len(stats['identifiers'])
        metadata['successful_count'] = stats['successful_count']
        metadata['failed_count'] = stats['failed_count']
        metadata['total_time_seconds'] = stats['total_time']
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        saved_files['metadata'] = str(meta_path)
        logger.info(f"Metadata saved to: {meta_path}")
        
        # 3. Save statistics
        stats_path = output_dir / f"{prefix}_stats.txt"
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"Protein Embedding Generation Statistics\n")
            f.write(f"========================================\n")
            f.write(f"Generation time: {metadata.get('timestamp', 'N/A')}\n")
            f.write(f"Model: {metadata.get('model', 'N/A')}\n")
            f.write(f"Device: {metadata.get('device', 'N/A')}\n")
            f.write(f"Total sequences processed: {len(stats['identifiers']) + stats['failed_count']}\n")
            f.write(f"Successfully processed: {stats['successful_count']}\n")
            f.write(f"Failed: {stats['failed_count']}\n")
            f.write(f"Total time: {stats['total_time']:.2f} seconds\n")
            f.write(f"Average time per protein: {stats['total_time']/max(1, stats['successful_count']):.2f} seconds\n")
            f.write(f"Embedding dimension: {metadata.get('embedding_dim', 'N/A')}\n")
            f.write(f"Pooling method: {metadata.get('pooling', 'N/A')}\n")
            f.write(f"Layers: {metadata.get('layers', [])}\n")
            
            if stats['sequence_lengths']:
                lengths = stats['sequence_lengths']
                f.write(f"\nSequence length statistics:\n")
                f.write(f"  Shortest: {min(lengths)} amino acids\n")
                f.write(f"  Longest: {max(lengths)} amino acids\n")
                f.write(f"  Average: {np.mean(lengths):.1f} amino acids\n")
                f.write(f"  Median: {np.median(lengths):.1f} amino acids\n")
            
            # Count .npy files
            npy_files = list(output_dir.glob("*.npy"))
            f.write(f"\nGenerated .npy files: {len(npy_files)}\n")
            f.write(f"Output directory: {output_dir}\n")
            
            if npy_files:
                f.write(f"\nExample .npy files:\n")
                for i, npy_file in enumerate(npy_files[:10]):
                    f.write(f"  {i+1}. {npy_file.name}\n")
                if len(npy_files) > 10:
                    f.write(f"  ... and {len(npy_files) - 10} more\n")
            
            if stats['failed_proteins']:
                f.write(f"\nFailed proteins ({len(stats['failed_proteins'])}):\n")
                for i, (protein_id, error) in enumerate(stats['failed_proteins'][:20]):
                    f.write(f"  {i+1}. {protein_id}: {error}\n")
                if len(stats['failed_proteins']) > 20:
                    f.write(f"  ... and {len(stats['failed_proteins']) - 20} more\n")
            
            f.write(f"\nSaved files:\n")
            f.write(f"  individual_npy_files: {len(npy_files)} .npy files\n")
            f.write(f"  labels_csv: {csv_path.name if 'labels_csv' in saved_files else 'None'}\n")
            f.write(f"  metadata: {meta_path.name}\n")
            f.write(f"  statistics: {stats_path.name}\n")
        
        saved_files['stats'] = str(stats_path)
        logger.info(f"Statistics saved to: {stats_path}")
        
        return saved_files


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Generate protein embeddings using local ESM model, one sequence at a time"
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
        help="ESM model file path"
    )
    
    parser.add_argument(
        "-p", "--pooling",
        default="none", # This is the key fix: defaults to residue-level features
        choices=["mean", "sum", "max", "cls", "last", "none"],
        help="Pooling method (default: none, which gives node-level 2D features for GNNs)"
    )
    
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Layers to extract (e.g., 6 12 24 36). Default is the last layer only."
    )
    
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "cuda:0", "cuda:1"],
        help="Compute device"
    )
    
    parser.add_argument(
        "--per_residue",
        action="store_true",
        help="Also save per-residue embeddings (not implemented in this version)"
    )
    
    parser.add_argument(
        "--prefix",
        default="protein_embeddings",
        help="Output file prefix for metadata and stats files (default: protein_embeddings)"
    )
    
    parser.add_argument(
        "--delimiter",
        default="\t",
        help="Delimiter for tabular format (default: tab)"
    )
    
    args = parser.parse_args()
    
    # Print parameters
    print("="*50)
    print("ESM Protein Embedding Generator (One by One)")
    print("="*50)
    print(f"Input file: {args.input_file}")
    print(f"Input format: {args.format}")
    print(f"Model file: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Pooling method: {args.pooling}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print("="*50)
    
    # Initialize embedder
    try:
        embedder = ESMProteinEmbedder(
            model_path=args.model_path,
            device=args.device
        )
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
    except Exception as e:
        print(f"Error reading sequences: {e}")
        return
    
    if not sequences:
        print("Error: No valid protein sequences were read")
        return
    
    if args.layers:
        layers = args.layers
    else:
        layers = [embedder.num_layers]
    
    # Generate embeddings one by one
    print("Generating embeddings one by one...")
    try:
        stats = embedder.generate_embeddings_one_by_one(
            sequences=sequences,
            output_dir=args.output_dir,
            layers=layers,
            pooling=args.pooling,
            include_per_residue=args.per_residue
        )
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return
    
    # Save metadata and statistics
    print("Saving metadata and statistics...")
    try:
        saved_files = embedder.save_metadata_and_stats(
            stats=stats,
            output_dir=args.output_dir,
            prefix=args.prefix
        )
    except Exception as e:
        print(f"Error saving metadata: {e}")
        return
    
    # Print completion information
    print("\n" + "="*50)
    print("Embedding generation completed successfully!")
    print("="*50)
    
    # Display saved files
    output_path = Path(args.output_dir)
    npy_files = list(output_path.glob("*.npy"))
    
    if npy_files:
        print(f"\nGenerated {len(npy_files)} individual .npy files:")
        for i, npy_file in enumerate(npy_files[:5]):
            try:
                data = np.load(npy_file)
                print(f"  {i+1}. {npy_file.name} - shape: {data.shape}")
            except:
                pass
        if len(npy_files) > 5:
            print(f"  ... and {len(npy_files) - 5} more")

if __name__ == "__main__":
    main()