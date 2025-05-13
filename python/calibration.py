import argparse
import math
import random
import sys
import time
import xxhash

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import torch_xla
from torch_xla.core.xla_model import xla_device
import torch_xla.core.xla_model as xm

from torch.utils.data import DataLoader, TensorDataset

from shapely.geometry import Polygon

from sklearn.model_selection import train_test_split, StratifiedKFold

from reader import CsvReader
from fast_geom import FastGeom
from candidate_stats import CandidateStats
from related_geoms import RelatedGeometries
from thresholds import (
    threshold_recall_wilson,
    threshold_recall_confidence,
    threshold_quant_ci,
)
from ensemble import (
    ensemble_threshold,
    ensemble_threshold_multi,
)




import math
import numpy as np
import random
import sys
import time
import pandas as pd

from shapely.geometry import Polygon

from sklearn.model_selection import train_test_split, StratifiedKFold

from scipy.special import logit, expit
from scipy.optimize import minimize
from scipy.stats import beta, norm

import xxhash

import torch
import torch.nn as nn
import torch.optim as optim

import torch_xla
from torch_xla.core.xla_model import xla_device
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.metrics as met

from torch.utils.data import DataLoader, TensorDataset


class calibration_Based_Algorithm:

    def __init__(self, budget: int, qPairs: int, delimiter: str, sourceFilePath: str, targetFilePath: str, target_recall, sampling_method, threshold_method):
        self.CLASS_SIZE = 500
        self.NO_OF_FEATURES = 16
        self.SAMPLE_SIZE = 50000
        self.POSITIVE_PAIR = 1
        self.NEGATIVE_PAIR = 0
        self.budget = budget
        self.target_recall = target_recall
        self.sampling_method = sampling_method
        self.threshold_method = threshold_method

        self.loader = CsvReader()
        print("→ Reading source geometries...")
        self.sourceData = self.loader.readAllEntities(sourceFilePath, delimiter)
        print(f"Loaded {len(self.sourceData)} source geometries.")

        print("→ Reading target geometries...")
        self.targetData = self.loader.readAllEntities(targetFilePath, delimiter)
        print(f"Loaded {len(self.targetData)} target geometries.")


        self.calibration_sample = set()
        self.predicted_probabilities = []
        self.relations = RelatedGeometries(qPairs)
        self.sample = set()
        self.verifiedPairs = set()
        self.totalCandidatePairs = {}
        self.geom_utils = FastGeom() # get number of coordinates
        self.cand_stats = CandidateStats() #get candidates

    def format_time(self, milliseconds):
        """Convert milliseconds to a human-readable string (hh:mm:ss)."""
        seconds = milliseconds / 1000
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02}:{minutes:02}:{seconds:06.3f}"  # hh:mm:ss.sss format

    def applyProcessing(self):
        time1 = int(time.time() * 1000)

        print("setThetas is starting")
        self.setThetas()

        time2 = int(time.time() * 1000)

        print("preprocessing is starting")
        self.preprocessing()
        time3 = int(time.time() * 1000)

        print("trainModel is starting")
        self.trainModel()
        time4 = int(time.time() * 1000)

        if self.sampling_method == "hashing":
          print("sampling is starting")
          self.build_calibration_sample()
          time5 = int(time.time() * 1000)
        else:
          time5 = time4

        print("verification is starting")
        self.verification()
        time6 = int(time.time() * 1000)

        print("Indexing Time\t:\t", self.format_time(time2 - time1))
        print("Initialization Time\t:\t", self.format_time(time3 - time2))
        print("Training Time\t:\t", self.format_time(time4 - time3))
        if self.sampling_method == "hashing":
          print("Sampling Time\t:\t", self.format_time(time5 - time4))
        print("Verification Time\t:\t", self.format_time(time6 - time5))

        self.relations.print()

    """
    def estimate_grid_size(self, n, factor=4, min_size=16, max_size=256):
        dim = int(np.ceil(np.sqrt(n)) / factor)
        return max(min_size, min(dim, max_size)), max(min_size, min(dim, max_size))
    """

    def build_candidate_csr(self, indices, num_targets):
        """
        Converts (sId, tId) pairs into CSR-like structure:
        offsets[t+1] - offsets[t] = number of sourceIds for targetId t
        values[offsets[t]:offsets[t+1]] = actual sourceIds for targetId t
        """
        sIds = indices[:, 0].astype(np.int32)
        tIds = indices[:, 1].astype(np.int32)

        # Sort by target ID
        sorted_idx = np.argsort(tIds)
        sIds_sorted = sIds[sorted_idx]
        tIds_sorted = tIds[sorted_idx]

        # Count how many sources per target
        unique_tids, counts = np.unique(tIds_sorted, return_counts=True)

        # Offset: 1 extra for last end index
        offsets = np.zeros(num_targets + 1, dtype=np.int32)
        offsets[unique_tids + 1] = counts
        offsets = np.cumsum(offsets)

        return offsets, sIds_sorted


    def estimate_grid_size(self, n, factor=4, min_size=16, max_size=256):
        dim = int(np.ceil(np.sqrt(n)) / factor)
        return dim, dim


    def preprocessing(self):

        def hashed_sample_ids(max_id, sample_size, seed, strata_id=None):
            """
            Deterministic, quick sampler that returns `sample_size` distinct ids
            from 0 … max_id-1.  If `strata_id` (array) is supplied, each id is
            hashed together with its stratum => stratified but still exact size.
            """
            ids = np.arange(max_id, dtype=np.int64)

            if strata_id is None:
                keys = np.fromiter(
                    (xxhash.xxh64(int(i).to_bytes(8, 'little'), seed=seed).intdigest()
                    for i in ids),
                    dtype=np.uint64, count=max_id
                )
            else:
                strata_id = np.asarray(strata_id, dtype=np.int64)
                if strata_id.size != max_id:
                    raise ValueError("strata_id length must equal max_id")
                keys = np.fromiter(
                    (xxhash.xxh64(
                        int(i).to_bytes(8, 'little') + int(s).to_bytes(8, 'little'),
                        seed=seed
                    ).intdigest()
                    for i, s in zip(ids, strata_id)),
                    dtype=np.uint64, count=max_id
                )

            # choose the ids with the `sample_size` smallest hash keys
            top_k = np.argpartition(keys, sample_size)[:sample_size]
            return set(ids[top_k])

        SEED = 2025
        max_candidates = 10 * len(self.sourceData)

        # for per‑target stratification, we make an array of targetId
        strata_id = np.arange(max_candidates) % len(self.targetData)

        self.sample_ids = hashed_sample_ids(max_id=max_candidates,
                                            sample_size=self.SAMPLE_SIZE,
                                            seed=SEED)


        if self.sampling_method == "random":
          self.calibration_sample_ids = set()
          while len(self.calibration_sample_ids) < 250000:
            self.calibration_sample_ids.add(random.randint(0, max_candidates))


        # 2) Initialize arrays
        self.flag = [-1] * len(self.sourceData)
        self.frequency = [-1] * len(self.sourceData)
        self.distinctCooccurrences = [0] * len(self.sourceData)
        self.realCandidates = [0] * len(self.sourceData)
        self.totalCooccurrences = [0] * len(self.sourceData)
        self.maxFeatures = [-sys.float_info.max] * self.NO_OF_FEATURES
        self.minFeatures = [sys.float_info.max] * self.NO_OF_FEATURES

        # Precompute bounding boxes for Target Data (we have already precomputed bounding boxes for Source Data in setThetas method)

        self.targetBounds = self.geom_utils.get_bounds(self.targetData)
        print("Computing Bounds finished")

        # Snap source bounds to theta grid
        src_grid_bounds = np.stack([
            np.floor(self.sourceBounds[:, 0] / self.thetaX),
            np.floor(self.sourceBounds[:, 1] / self.thetaY),
            np.ceil(self.sourceBounds[:, 2] / self.thetaX),
            np.ceil(self.sourceBounds[:, 3] / self.thetaY),
        ], axis=1).astype(np.float32)

        # Snap target bounds to same grid
        tgt_grid_bounds = np.stack([
            np.floor(self.targetBounds[:, 0] / self.thetaX),
            np.floor(self.targetBounds[:, 1] / self.thetaY),
            np.ceil(self.targetBounds[:, 2] / self.thetaX),
            np.ceil(self.targetBounds[:, 3] / self.thetaY),
        ], axis=1).astype(np.float32)

        minx = min(src_grid_bounds[:, 0].min(), tgt_grid_bounds[:, 0].min())
        miny = min(src_grid_bounds[:, 1].min(), tgt_grid_bounds[:, 1].min())
        maxx = max(src_grid_bounds[:, 2].max(), tgt_grid_bounds[:, 2].max())
        maxy = max(src_grid_bounds[:, 3].max(), tgt_grid_bounds[:, 3].max())
        extent = (minx, miny, maxx, maxy)

        grid_x, grid_y = self.estimate_grid_size(max(len(self.sourceBounds), len(self.targetBounds)))


        indices = self.geom_utils.grid_bbox_intersect(
            src_grid_bounds,
            tgt_grid_bounds,
            extent,
            grid_x,
            grid_y
        )

        all_tgt_ids = indices[:, 1]
        all_src_ids = indices[:, 0]

        # Build CSR candidate structure
        self.candidate_offsets, self.candidate_values = self.build_candidate_csr(indices, len(self.targetData))

        self.flag = [-1] * len(self.sourceData)
        self.frequency = [0] * len(self.sourceData)

        for sId, tId in indices:
            if self.flag[sId] != tId:
                self.flag[sId] = tId
                self.frequency[sId] = 0
            self.frequency[sId] += 1


        print("all indices are",len(indices))

        # Compute Areas
        self.sourceDataAreas = (self.sourceBounds[:, 2] - self.sourceBounds[:, 0]) * (self.sourceBounds[:, 3] - self.sourceBounds[:, 1])
        self.targetDataAreas = (self.targetBounds[:, 2] - self.targetBounds[:, 0]) * (self.targetBounds[:, 3] - self.targetBounds[:, 1])

        print("Computing Areas finished")

        # Compute Blocks
        self.SourceBlocks = self.getNoOfBlocks1(self.sourceBounds)
        self.TargetBlocks = self.getNoOfBlocks1(self.targetBounds)

        print("Computing Blocks finished")

        # Compute Number of Points
        self.source_no_of_points = self.geom_utils.get_num_of_points(self.sourceData)
        self.target_no_of_points = self.geom_utils.get_num_of_points(self.targetData)

        print("Computing No of Points finished")

        # Computes Lengths
        self.SourceGeomLength = self.geom_utils.get_lengths(self.sourceData)
        self.targetGeomLength = self.geom_utils.get_lengths(self.targetData)

        print("Computing Length finished")

        all_pair_ids  = []  # We'll track the pairId so we can do sample/calibration logic after the bulk check

        #pairId = 0
        for targetId in range(len(self.targetData)):
            start = self.candidate_offsets[targetId]
            end = self.candidate_offsets[targetId + 1]
            candidateMatches = self.candidate_values[start:end]

            currentCooccurrences = 0

            # Update co-occurrence stats
            for candidateMatchId in candidateMatches:
                self.distinctCooccurrences[candidateMatchId] += 1

                # Accumulate frequency for this target
                currentCooccurrences += self.frequency[candidateMatchId]

                self.totalCooccurrences[candidateMatchId] += self.frequency[candidateMatchId]

            currentDistinctCooccurrences = len(candidateMatches)

            # Update min/max currentCooccurrences and currentDistinctCooccurrences
            self.maxFeatures[13] = max(self.maxFeatures[13], currentCooccurrences)
            self.minFeatures[13] = min(self.minFeatures[13], currentCooccurrences)

            self.maxFeatures[14] = max(self.maxFeatures[14], currentDistinctCooccurrences)
            self.minFeatures[14] = min(self.minFeatures[14], currentDistinctCooccurrences)

        # --- Convert lists to NumPy arrays ---
        all_src_boxes = self.sourceBounds[all_src_ids]  # shape (N,4)
        all_tgt_boxes = self.targetBounds[all_tgt_ids]  # shape (N,4)

        # --- Vectorized bounding-box intersection (including edge contact) ---
        minxA = all_src_boxes[:, 0]
        minyA = all_src_boxes[:, 1]
        maxxA = all_src_boxes[:, 2]
        maxyA = all_src_boxes[:, 3]

        minxB = all_tgt_boxes[:, 0]
        minyB = all_tgt_boxes[:, 1]
        maxxB = all_tgt_boxes[:, 2]
        maxyB = all_tgt_boxes[:, 3]

        intersect_w = np.minimum(maxxA, maxxB) - np.maximum(minxA, minxB)
        intersect_h = np.minimum(maxyA, maxyB) - np.maximum(minyA, minyB)

        # "Touch" => >= 0
        intersect_mask = (intersect_w >= 0) & (intersect_h >= 0)

        # intersection area for min/max stats
        clipped_w = np.clip(intersect_w, 0, None)
        clipped_h = np.clip(intersect_h, 0, None)
        mbr_intersection = clipped_w * clipped_h

        # --- Filter only the valid pairs ---
        valid_src_ids  = all_src_ids[intersect_mask]
        valid_tgt_ids  = all_tgt_ids[intersect_mask]
        valid_pair_ids = all_tgt_ids[intersect_mask]

        # We'll sort by targetId ascending (optional)
        sort_idx = np.argsort(valid_tgt_ids)
        valid_src_ids  = valid_src_ids[sort_idx]
        valid_tgt_ids  = valid_tgt_ids[sort_idx]
        valid_pair_ids = valid_pair_ids[sort_idx]

        # Store them if needed
        self.filtered_source_ids = valid_src_ids.tolist()
        self.filtered_target_ids = valid_tgt_ids.tolist()

        currentCandidatesArray = [0]*len(self.targetData)  # for each targetId

        # For each *valid* pair:
        for i, pid in enumerate(valid_pair_ids):
            sId = valid_src_ids[i]
            tId = valid_tgt_ids[i]

            # Because we know it intersects, we treat that as "real candidate"
            self.realCandidates[sId] += 1

            # "currentCandidates" means: how many valid candidates does each target have
            currentCandidatesArray[tId] += 1

            # Check if pairId is in sample_ids or calibration_sample_ids
            if pid in self.sample_ids:
                self.sample.add((sId, tId))
            if (self.sampling_method == 'random') and (pid in self.calibration_sample_ids):
                self.calibration_sample.add((sId, tId))

        # Now we have a count of how many valid (intersecting) candidates each target has
        max_cc = max(currentCandidatesArray)
        min_cc = min(currentCandidatesArray)
        self.maxFeatures[15] = max(self.maxFeatures[15], max_cc)
        self.minFeatures[15] = min(self.minFeatures[15], min_cc)

        # Update min/max features for the intersection area, etc. ---
        valid_mbr_intersection = mbr_intersection[intersect_mask]
        if len(valid_mbr_intersection) > 0:
            pos_areas = valid_mbr_intersection[valid_mbr_intersection > 0]
            if len(pos_areas) > 0:
                self.maxFeatures[2] = max(self.maxFeatures[2], pos_areas.max())
                self.minFeatures[2] = min(self.minFeatures[2], pos_areas.min())

        # Also update the rest of the min/max features:
        self.maxFeatures[0] = max(self.maxFeatures[0], self.sourceDataAreas.max())
        self.minFeatures[0] = min(self.minFeatures[0], self.sourceDataAreas.min())

        self.maxFeatures[1] = max(self.maxFeatures[1], self.targetDataAreas.max())
        self.minFeatures[1] = min(self.minFeatures[1], self.targetDataAreas.min())

        self.maxFeatures[3] = max(self.maxFeatures[3], max(self.SourceBlocks))
        self.minFeatures[3] = min(self.minFeatures[3], min(self.SourceBlocks))

        self.maxFeatures[4] = max(self.maxFeatures[4], max(self.TargetBlocks))
        self.minFeatures[4] = min(self.minFeatures[4], min(self.TargetBlocks))

        self.maxFeatures[5] = max(self.maxFeatures[5], max(self.frequency))
        self.minFeatures[5] = min(self.minFeatures[5], min(self.frequency))

        self.maxFeatures[6] = max(self.maxFeatures[6],self.source_no_of_points.max())
        self.minFeatures[6] = min(self.minFeatures[6], self.source_no_of_points.min())

        self.maxFeatures[7] = max(self.maxFeatures[7], self.target_no_of_points.max())
        self.minFeatures[7] = min(self.minFeatures[7], self.target_no_of_points.min())

        self.maxFeatures[8] = max(self.maxFeatures[8], self.SourceGeomLength.max())
        self.minFeatures[8] = min(self.minFeatures[8], self.SourceGeomLength.min())

        self.maxFeatures[9] = max(self.maxFeatures[9], self.targetGeomLength.max())
        self.minFeatures[9] = min(self.minFeatures[9], self.targetGeomLength.min())

        self.maxFeatures[10] = max(self.totalCooccurrences)
        self.minFeatures[10] = min(self.totalCooccurrences)

        self.maxFeatures[11] = max(self.distinctCooccurrences)
        self.minFeatures[11] = min(self.distinctCooccurrences)

        self.maxFeatures[12] = max(self.realCandidates)
        self.minFeatures[12] = min(self.realCandidates)


    def build_calibration_sample(self, max_pairs=250_000, seed=2026):
        """
        1.  Score *all* filtered pairs with the model already trained in
            trainModel().
        2.  Put each pair into a score decile (0‑9).
        3.  Deterministically select `max_pairs` ids, stratified by decile.
        4.  Fill `self.calibration_sample` with those (sId, tId) pairs.
        """
        # ===  vector scores for EVERY candidate pair =================
        feats   = self.get_feature_vector(self.filtered_source_ids,
                                          self.filtered_target_ids)
        scores  = self.predict_in_batches(feats, batch_size=8192).ravel()

        import xxhash, numpy as np

        # ===  decile for each pair ===================================
        deciles = np.minimum(9, (scores * 10).astype(int))

        # === deterministic hash sample, length = len(filtered pairs) ==

        Npairs = len(self.filtered_source_ids)
        ids    = np.arange(Npairs, dtype=np.int64)

        keys = np.fromiter(
            (xxhash.xxh64(
                ids[i].tobytes() + deciles[i].tobytes(),
                seed=seed).intdigest() for i in range(Npairs)),
            dtype=np.uint64, count=Npairs
        )
        k = min(max_pairs, Npairs)
        sel_idx = np.argpartition(keys, k)[:k]        # chosen indices

        # === rebuild self.calibration_sample =================================
        self.calibration_sample.clear()
        for idx in sel_idx:
            sId = self.filtered_source_ids[idx]
            tId = self.filtered_target_ids[idx]
            self.calibration_sample.add((sId, tId))

        self.relations.reset()
        print(f"[build_calibration_sample] selected {len(self.calibration_sample)} pairs "
              f"(stratified by score decile)")


    def setThetas(self):

         #We compute Bounds for Source Data
        self.sourceBounds = self.geom_utils.get_bounds(self.sourceData)  # shape (N, 4)
        print(self.sourceBounds)

        # Each row: [minx, miny, maxx, maxy]
        # Summation approach
        self.thetaX = 0.0
        self.thetaY = 0.0
        for env in self.sourceBounds:
            self.thetaX += (env[2] - env[0])  # maxx - minx
            self.thetaY += (env[3] - env[1])  # maxy - miny

        N = len(self.sourceData)
        self.thetaX /= N
        self.thetaY /= N
        print("Dimensions of Equigrid:", self.thetaX, self.thetaY)



    @staticmethod
    def create_model(input_dim):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = nn.Linear(input_dim, 128)
                self.dropout1 = nn.Dropout(0.3)
                self.bn1 = nn.BatchNorm1d(128)
                self.fc2 = nn.Linear(128, 64)
                self.dropout2 = nn.Dropout(0.5)
                self.bn2 = nn.BatchNorm1d(64)
                self.fc3 = nn.Linear(64, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout1(x)
                x = self.bn1(x)
                x = torch.relu(self.fc2(x))
                x = self.dropout2(x)
                x = self.bn2(x)
                x = torch.sigmoid(self.fc3(x))
                return x

        return Model()



    def predict(self, X):
        # Predict using the trained model
        self.classifier.eval()
        device = xla_device()  # TPU device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            predictions = self.classifier(X_tensor)
        return predictions.cpu().numpy()  # Move predictions to CPU and return as NumPy array


    def predict_in_batches(self, X, batch_size=8192):
        self.classifier.eval()
        device = xla_device()

        X = np.asarray(X, dtype=np.float32)
        outputs = []

        for start_idx in range(0, len(X), batch_size):
            end_idx = start_idx + batch_size
            X_batch = X[start_idx:end_idx]

            X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)

            with torch.no_grad():
                preds = self.classifier(X_tensor)

            outputs.append(preds.cpu().numpy())
            xm.mark_step()  # Important on TPU

        return np.concatenate(outputs, axis=0)



    def validate_data(self, X, y):
        # Ensure both positive and negative classes are represented
        if np.count_nonzero(y == 0) == 0 or np.count_nonzero(y == 1) == 0:
            raise ValueError("Both negative and positive instances must be labeled.")

        # Validate X
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            raise ValueError("X must be a 2D NumPy array, but got: {}".format(X))

        return X, y

    def train_model(self, X, y):
        # Validate the input data
        X, y = self.validate_data(X, y)

        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        # Create dataset and dataloaders
        dataset = TensorDataset(X_tensor, y_tensor)
        train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)

        train_loader = DataLoader(
            train_data, batch_size=128, shuffle=True, num_workers=0, persistent_workers=False
        )
        val_loader = DataLoader(
            val_data, batch_size=128, shuffle=False, num_workers=0, persistent_workers=False
        )

        # Get model and move it to TPU
        device = xm.xla_device()
        model = self.create_model(X.shape[1]).to(device)

        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        epochs = 30
        patience = 3
        best_model_state = None  # Store best model

        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                xm.optimizer_step(optimizer)  # TPU-specific optimizer step
                xm.mark_step()  # Ensures TPU execution

                running_loss += loss.item()

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()  # Save best model
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load and store the best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        self.classifier = model  # Store trained model



    def trainModel(self):
        # Shuffle your sample of pairs
        self.sample = list(self.sample)
        random.shuffle(self.sample)

        # Separate out (geomIds1, geomIds2) from self.sample
        geomIds1 = []
        geomIds2 = []
        for (sourceId, targetId) in self.sample:
            geomIds1.append(sourceId)
            geomIds2.append(targetId)

        # Collect the actual geometries (in the *same order*)
        sourceGeoms = [self.sourceData[sid] for sid in geomIds1]
        targetGeoms = [self.targetData[tid] for tid in geomIds2]

        # Call the *vectorized* method to do all relationship checks at once
        related_mask = self.relations.verifyRelations(geomIds1, geomIds2, sourceGeoms, targetGeoms)

        # Now pick exactly CLASS_SIZE positive and negative pairs
        negativePairs, positivePairs, excessVerifications = 0, 0, 0
        SourceIds, TargetIds, y = [], [], []

        excesive_positive_sourceIds = []
        excesive_positive_targetIds = []

        # We iterate in the *same order* as self.sample,
        # but rely on related_mask[i] to know if it's related
        for i, (sourceId, targetId) in enumerate(self.sample):
            if negativePairs == self.CLASS_SIZE and positivePairs == self.CLASS_SIZE:
                break

            # Mark that we've verified this pair
            self.verifiedPairs.add((sourceId, targetId))

            isRelated = related_mask[i]

            if isRelated:
                if positivePairs < self.CLASS_SIZE:
                    positivePairs += 1
                    y.append(self.POSITIVE_PAIR)
                    SourceIds.append(sourceId)
                    TargetIds.append(targetId)
                else:
                    excessVerifications += 1
                    excesive_positive_sourceIds.append(sourceId)
                    excesive_positive_targetIds.append(targetId)
            else:
                if negativePairs < self.CLASS_SIZE:
                    negativePairs += 1
                    y.append(self.NEGATIVE_PAIR)
                    SourceIds.append(sourceId)
                    TargetIds.append(targetId)
                else:
                    excessVerifications += 1

        X = self.get_feature_vector(SourceIds, TargetIds)

        print("Positive Verifications\t:\t" + str(positivePairs))
        print("Negative Verifications\t:\t" + str(negativePairs))
        print("Excess Verifications\t:\t" + str(excessVerifications))


        y = np.array(y)
        if negativePairs == 0 or positivePairs == 0:
            raise ValueError("Both negative and positive instances must be labelled.")

        # ... Train our model ...
        input_dim = X.shape[1]
        model = self.create_model(input_dim)  # create a classifier
        self.train_model(X, y)               # do our actual training


    def get_feature_vector(self, sourceIds, targetIds):
        """
        Construct feature vectors for each (sourceId, targetId) pair
        using precomputed bounding boxes, areas, lengths, etc.
        """

        # Number of pairs
        N = len(sourceIds)
        featureVectors = np.zeros((N, self.NO_OF_FEATURES), dtype=np.float32)

        # Build NumPy arrays of sourceIds, targetIds
        sourceIds = np.asarray(sourceIds, dtype=np.int32)
        targetIds = np.asarray(targetIds, dtype=np.int32)

        # MBR intersection (bounding-box intersection) in vectorized form
        #    - We have self.sourceBounds[i] = [minx, miny, maxx, maxy] for each source geometry i
        #    - same for self.targetBounds
        s_bounds = self.sourceBounds[sourceIds]  # shape (N,4)
        t_bounds = self.targetBounds[targetIds]  # shape (N,4)

        s_minx = s_bounds[:, 0]
        s_miny = s_bounds[:, 1]
        s_maxx = s_bounds[:, 2]
        s_maxy = s_bounds[:, 3]

        t_minx = t_bounds[:, 0]
        t_miny = t_bounds[:, 1]
        t_maxx = t_bounds[:, 2]
        t_maxy = t_bounds[:, 3]

        intersect_w = np.minimum(s_maxx, t_maxx) - np.maximum(s_minx, t_minx)
        intersect_h = np.minimum(s_maxy, t_maxy) - np.maximum(s_miny, t_miny)
        clipped_w = np.clip(intersect_w, 0, None)
        clipped_h = np.clip(intersect_h, 0, None)

        mbr_intersection = clipped_w * clipped_h  # shape (N,)

        # Vector‐retrieve precomputed geometry features
        # areas
        s_areas = self.sourceDataAreas[sourceIds]
        t_areas = self.targetDataAreas[targetIds]

        # lengths
        s_length = self.SourceGeomLength[sourceIds]
        t_length = self.targetGeomLength[targetIds]

        # blocks
        s_blocks = np.array(self.SourceBlocks) # Convert to NumPy array
        t_blocks = np.array(self.TargetBlocks) # Convert to NumPy array
        s_blocks = s_blocks[sourceIds]
        t_blocks = t_blocks[targetIds]

        # boundary points
        s_npoints = self.source_no_of_points[sourceIds]
        t_npoints = self.target_no_of_points[targetIds]

        # “valid_intersections”
        # For each target, how many pairs have intersection>0?
        # We do the grouping by targetId in a vectorized manner:
        intersection_binary = (mbr_intersection > 0).astype(np.float32)

        # group by targetIds
        unique_targets, inv_idx = np.unique(targetIds, return_inverse=True)
        sum_intersect_by_target = np.bincount(inv_idx, weights=intersection_binary)
        valid_intersections = sum_intersect_by_target[inv_idx]  # shape (N,)

        # Candidate-based features require dictionary lookups
        #    totalCandidatePairs[targetId] => a list of candidateMatches
        freq_sums = np.zeros(N, dtype=np.float32)
        lens = np.zeros(N, dtype=np.float32)

        freq_sums, lens = self.cand_stats.compute(
            targetIds,
            self.candidate_offsets,
            self.candidate_values,
            self.frequency
        )

        # Fill the featureVectors with partial vectorization
        # feature [0] => source area
        featureVectors[:, 0] = (s_areas - self.minFeatures[0]) / (self.maxFeatures[0]) * 10000
        # feature [1] => target area
        featureVectors[:, 1] = (t_areas - self.minFeatures[1]) / (self.maxFeatures[1]) * 10000
        # feature [2] => intersection area
        featureVectors[:, 2] = (mbr_intersection - self.minFeatures[2]) / (self.maxFeatures[2]) * 10000

        # feature [3] => source blocks
        featureVectors[:, 3] = (s_blocks - self.minFeatures[3]) / (self.maxFeatures[3]) * 10000
        # feature [4] => target blocks
        featureVectors[:, 4] = (t_blocks - self.minFeatures[4]) / (self.maxFeatures[4]) * 10000
        # feature [5] => "common blocks" or frequency[sourceId]
        freq_src = np.asarray(self.frequency)[sourceIds]  # shape (N,)
        featureVectors[:, 5] = (freq_src - self.minFeatures[5]) / (self.maxFeatures[5]) * 10000

        # feature [6] => source boundary points
        featureVectors[:, 6] = (s_npoints - self.minFeatures[6]) / (self.maxFeatures[6]) * 10000
        # feature [7] => target boundary points
        featureVectors[:, 7] = (t_npoints - self.minFeatures[7]) / (self.maxFeatures[7]) * 10000
        # feature [8] => source length
        featureVectors[:, 8] = (s_length - self.minFeatures[8]) / (self.maxFeatures[8]) * 10000
        # feature [9] => target length
        featureVectors[:, 9] = (t_length - self.minFeatures[9]) / (self.maxFeatures[9]) * 10000

        # feature [10] => self.totalCooccurrences[sourceId]
        totalCooc_src = np.asarray(self.totalCooccurrences)[sourceIds]
        featureVectors[:, 10] = (totalCooc_src - self.minFeatures[10]) / (self.maxFeatures[10]) * 10000

        # feature [11] => self.distinctCooccurrences[sourceId]
        distinctCooc_src = np.asarray(self.distinctCooccurrences)[sourceIds]
        featureVectors[:, 11] = (distinctCooc_src - self.minFeatures[11]) / (self.maxFeatures[11]) * 10000

        # feature [12] => self.realCandidates[sourceId]
        realCands_src = np.asarray(self.realCandidates)[sourceIds]
        featureVectors[:, 12] = (realCands_src - self.minFeatures[12]) / (self.maxFeatures[12]) * 10000

        # feature [13] => sum of frequency for candidateMatches
        featureVectors[:, 13] = (freq_sums - self.minFeatures[13]) / (self.maxFeatures[13]) * 10000

        # feature [14] => number of candidateMatches
        featureVectors[:, 14] = (lens - self.minFeatures[14]) / (self.maxFeatures[14]) * 10000

        # feature [15] => number of geometry intersections for that target (like "valid_intersections")
        featureVectors[:, 15] = (valid_intersections - self.minFeatures[15]) / (self.maxFeatures[15]) * 10000

        return featureVectors

    def getNoOfBlocks1(self, envelopes):
        blocks = []
        for envelope in envelopes:
            maxX = math.ceil(envelope[2] / self.thetaX)
            maxY = math.ceil(envelope[3] / self.thetaY)
            minX = math.floor(envelope[0] / self.thetaX)
            minY = math.floor(envelope[1] / self.thetaY)
            blocks.append((maxX - minX + 1) * (maxY - minY + 1))
        return blocks


    def verification(self):
        import random
        import numpy as np

        # Make sure we only do these conversions once (e.g. in __init__)
        if not hasattr(self, 'source_data_array'):
            self.source_data_array = np.array(self.sourceData, dtype=object)
        if not hasattr(self, 'target_data_array'):
            self.target_data_array = np.array(self.targetData, dtype=object)

        total_decisions, true_positive_decisions = len(self.verifiedPairs), 0

        # Shuffle the calibration sample
        self.calibration_sample = list(self.calibration_sample)
        random.shuffle(self.calibration_sample)


        # 1) Filter out pairs
        filteredGeomIds1 = []
        filteredGeomIds2 = []
        filteredSourceGeoms = []
        filteredTargetGeoms = []
        All_SourceInstanceIndexes = []
        All_TargetInstanceIndexes = []
        index_map = {}  # Maps original index in calibration_sample to index in filtered list

        for i, (sourceGeomId, targetGeomId) in enumerate(self.calibration_sample):
            # Always collect all indexes
            All_SourceInstanceIndexes.append(sourceGeomId)
            All_TargetInstanceIndexes.append(targetGeomId)

            # Only process unverified pairs
            if (sourceGeomId, targetGeomId) not in self.verifiedPairs:
                index_map[i] = len(filteredGeomIds1)
                total_decisions += 1
                filteredGeomIds1.append(sourceGeomId)
                filteredGeomIds2.append(targetGeomId)
                filteredSourceGeoms.append(self.sourceData[sourceGeomId])
                filteredTargetGeoms.append(self.targetData[targetGeomId])

        # Bulk verify the unverified pairs
        related_mask = self.relations.verifyRelations(
            filteredGeomIds1,
            filteredGeomIds2,
            filteredSourceGeoms,
            filteredTargetGeoms
        )

        # Initialize predicted labels with 0
        self.All_predicted_labels = [0] * len(self.calibration_sample)
        SourceInstanceIndexes = []
        TargetInstanceIndexes = []

        if self.threshold_method == 'QuantCI':  #In this case we want all the predicted probabilities, not only the ones labeled as positive
            # Build arrays and assign labels for related pairs
            for original_idx, filtered_idx in index_map.items():
                if related_mask[filtered_idx]:
                    sourceGeomId, targetGeomId = self.calibration_sample[original_idx]
                    SourceInstanceIndexes.append(sourceGeomId)
                    TargetInstanceIndexes.append(targetGeomId)
                    self.All_predicted_labels[original_idx] = 1

            # Extract features & Predict in batches
            All_Instances = self.get_feature_vector(All_SourceInstanceIndexes, All_TargetInstanceIndexes)
            self.All_predicted_probabilities = self.predict_in_batches(All_Instances, batch_size=8192).ravel()
            print('calibration verifications', total_decisions - len(self.verifiedPairs))
            print('calibration positive verifications', self.All_predicted_labels.count(1))


        else:     #In these cases we want only the predicted probabilities labeled as positive
            # Build arrays for the pairs that are related
            for idx, is_related in enumerate(related_mask):
                if is_related:
                    SourceInstanceIndexes.append(filteredGeomIds1[idx])
                    TargetInstanceIndexes.append(filteredGeomIds2[idx])


            # Extract features & Predict in batches
            Instances = self.get_feature_vector(SourceInstanceIndexes, TargetInstanceIndexes)
            self.predicted_probabilities = self.predict_in_batches(Instances, batch_size=8192).ravel()
            print('calibration verifications', total_decisions - len(self.verifiedPairs))
            print('calibration positive verifications', len(self.predicted_probabilities))

        self.relations.reset()


        if self.threshold_method == 'QuantCI':
            self.minimum_probability_threshold = threshold_quant_ci(self.All_predicted_probabilities, self.All_predicted_labels, self.target_recall)
            print("Minimum probability threshold", self.minimum_probability_threshold)


        elif self.threshold_method == 'Clopper_Pearson':
            self.minimum_probability_threshold = threshold_recall_confidence(
                self.predicted_probabilities,
                target_recall=self.target_recall      # e.g. 0 .90
            )
            print("Minimum probability threshold", self.minimum_probability_threshold)


        elif self.threshold_method == 'wilson':
            self.minimum_probability_threshold = threshold_recall_wilson(
                self.predicted_probabilities,
                target_recall=self.target_recall,      # e.g. 0 .90
                alpha=0.05
            )
            print("Minimum probability threshold", self.minimum_probability_threshold)


        elif self.threshold_method == 'ensemble':
            self.minimum_probability_threshold = ensemble_threshold(
                self.predicted_probabilities,
                target_recall=self.target_recall,   # 0.90
                alpha        =0.05,
                n_boot       =200,
                random_state =42,
                verbose      =True
            )
            print("Minimum probability threshold", self.minimum_probability_threshold)



        elif self.threshold_method == 'ensemble_multi':
            self.minimum_probability_threshold, debug_thresholds = ensemble_threshold_multi(
              self.predicted_probabilities,
              target_recall = self.target_recall,   # 0.90
              K            = 9,
              subsample_fr = 0.8,
              n_boot       = 200,
              verbose      = True
            )
            print("Minimum probability threshold", self.minimum_probability_threshold)

        SourceInstanceIndexes.clear()
        TargetInstanceIndexes.clear()

        SourceInstanceIndexes = self.filtered_source_ids
        TargetInstanceIndexes = self.filtered_target_ids

        # Convert them to NumPy arrays if you haven't already:
        SourceInstanceIndexes = np.array(SourceInstanceIndexes, dtype=np.int64)
        TargetInstanceIndexes = np.array(TargetInstanceIndexes, dtype=np.int64)

        # Create a mask that excludes pairs in self.verifiedPairs
        not_verified_mask = []
        for s, t in zip(SourceInstanceIndexes, TargetInstanceIndexes):
            not_verified_mask.append((s, t) not in self.verifiedPairs)

        not_verified_mask = np.array(not_verified_mask, dtype=bool)

        # Apply the mask
        SourceInstanceIndexes = SourceInstanceIndexes[not_verified_mask]
        TargetInstanceIndexes = TargetInstanceIndexes[not_verified_mask]

        print("SourceInstanceIndexes", len(SourceInstanceIndexes))
        print("TargetInstanceIndexes", len(TargetInstanceIndexes))

        #SourceInstanceIndexes = np.array(SourceInstanceIndexes)
        #TargetInstanceIndexes = np.array(TargetInstanceIndexes)

        # 6) Predict in batches again
        Instances = self.get_feature_vector(SourceInstanceIndexes, TargetInstanceIndexes)
        start = time.time()
        probabilities = self.predict_in_batches(Instances, batch_size=8192).ravel()
        print("Predict done in:", time.time() - start)

        start = time.time()
        valid_indexes = np.where(probabilities >= self.minimum_probability_threshold)[0]
        valid_indexes = valid_indexes[:self.budget]
        print("Index filtering done in:", time.time() - start)

        start = time.time()
        valid_source_indexes = SourceInstanceIndexes[valid_indexes]
        valid_target_indexes = TargetInstanceIndexes[valid_indexes]
        print("Index selection done in:", time.time() - start)

        start = time.time()
        valid_source_data = self.source_data_array[valid_source_indexes]
        valid_target_data = self.target_data_array[valid_target_indexes]
        print("Final data gathering done in:", time.time() - start)

        start = time.time()
        print("Verifying Relations")
        true_positive_decisions = self.relations.verifyRelations(
            valid_source_indexes,
            valid_target_indexes,
            valid_source_data,
            valid_target_data
        )
        print("Relation verification done in:", time.time() - start)
        print("True Positive Decisions\t:\t" + str(true_positive_decisions))

