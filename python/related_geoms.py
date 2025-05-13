from fast_geom import FastGeom
from de9im_patterns import (
    encode_de9im_strings,
    contains, crosses_lines, crosses_1, crosses_2,
    disjoint, equal, intersects_de9im,
    overlaps1, overlaps2,
    touches, within, covered_by, covers,
)
# inside verifyRelations():
import numpy as np
from shapely.wkb import dumps as wkb_dumps


class RelatedGeometries :
        def __init__(self, qualifyingPairs) :
            self.pgr = 0
            self.exceptions = 0
            self.detectedLinks = 0
            self.verifiedPairs = 0
            self.qualifyingPairs = qualifyingPairs
            self.interlinkedGeometries = 0
            self.continuous_unrelated_Pairs = 0
            self.violations = 0
            self.containsD1 = []
            self.containsD2 = []
            self.coveredByD1 = []
            self.coveredByD2 = []
            self.coversD1 = []
            self.coversD2 = []
            self.crossesD1 = []
            self.crossesD2 = []
            self.equalsD1 = []
            self.equalsD2 = []
            self.intersectsD1 = []
            self.intersectsD2 = []
            self.overlapsD1 = []
            self.overlapsD2 = []
            self.touchesD1 = []
            self.touchesD2 = []
            self.withinD1 = []
            self.withinD2 = []
            self.geom_utils = FastGeom()


        def addContains(self, gId1,  gId2) :
          self.containsD1.append(gId1)
          self.containsD2.append(gId2)
        def addCoveredBy(self, gId1,  gId2):
           self.coveredByD1.append(gId1)
           self.coveredByD2.append(gId2)
        def addCovers(self, gId1,  gId2):
           self.coversD1.append(gId1)
           self.coversD2.append(gId2)
        def addCrosses(self, gId1,  gId2) :
          self.crossesD1.append(gId1)
          self.crossesD2.append(gId2)
        def addEquals(self, gId1,  gId2) :
          self.equalsD1.append(gId1)
          self.equalsD2.append(gId2)
        def addIntersects(self, gId1,  gId2) :
          self.intersectsD1.append(gId1)
          self.intersectsD2.append(gId2)
        def addOverlaps(self, gId1,  gId2) :
          self.overlapsD1.append(gId1)
          self.overlapsD2.append(gId2)
        def addTouches(self, gId1,  gId2) :
          self.touchesD1.append(gId1)
          self.touchesD2.append(gId2)
        def addWithin(self, gId1,  gId2) :
          self.withinD1.append(gId1)
          self.withinD2.append(gId2)

        def  getInterlinkedPairs(self) :
            return self.interlinkedGeometries
        def  getNoOfContains(self) :
            return len(self.containsD1)
        def  getNoOfCoveredBy(self) :
            return len(self.coveredByD1)
        def  getNoOfCovers(self) :
            return len(self.coversD1)
        def  getNoOfCrosses(self) :
            return len(self.crossesD1)
        def  getNoOfEquals(self) :
            return len(self.equalsD1)
        def  getNoOfIntersects(self) :
            return len(self.intersectsD1)
        def  getNoOfOverlaps(self) :
            return len(self.overlapsD1)
        def  getNoOfTouches(self) :
            return len(self.touchesD1)
        def  getNoOfWithin(self) :
            return len(self.withinD1)
        def  getVerifiedPairs(self) :
            return self.verifiedPairs

        def reset(self):
            self.pgr = 0
            self.exceptions = 0
            self.detectedLinks = 0
            self.verifiedPairs = 0
            self.interlinkedGeometries = 0

            self.containsD1.clear()
            self.containsD2.clear()
            self.coveredByD1.clear()
            self.coveredByD2.clear()
            self.coversD1.clear()
            self.coversD2.clear()
            self.crossesD1.clear()
            self.crossesD2.clear()
            self.equalsD1.clear()
            self.equalsD2.clear()
            self.intersectsD1.clear()
            self.intersectsD2.clear()
            self.overlapsD1.clear()
            self.overlapsD2.clear()
            self.touchesD1.clear()
            self.touchesD2.clear()
            self.withinD1.clear()
            self.withinD2.clear()


        def print(self) :
            print("Qualifying pairs:\t", str(self.qualifyingPairs))
            print("Exceptions:\t", str(self.exceptions))
            print("Detected Links:\t", str(self.detectedLinks))
            print("Interlinked geometries:\t", str(self.interlinkedGeometries))
            print("No of contains:\t", str(self.getNoOfContains()))
            print("No of covered-by:\t" + str(self.getNoOfCoveredBy()))
            print("No of covers:\t", str(self.getNoOfCovers()))
            print("No of crosses:\t", str(self.getNoOfCrosses()))
            print("No of equals:\t", str(self.getNoOfEquals()))
            print("No of intersects:\t" + str(self.getNoOfIntersects()))
            print("No of overlaps:\t", str(self.getNoOfOverlaps()))
            print("No of touches:\t", str(self.getNoOfTouches()))
            print("No of within:\t", str(self.getNoOfWithin()))

            if self.qualifyingPairs != 0:
              print("Recall", str((self.interlinkedGeometries / float(self.qualifyingPairs))))
            else:
              print('array is empty')
            if self.verifiedPairs != 0:
              print("Precision", str((self.interlinkedGeometries / self.verifiedPairs)))
            else:
              print('array is empty 2')
            if self.qualifyingPairs != 0 and self.verifiedPairs != 0:
              print("Progressive Geometry Recall", str(self.pgr / self.qualifyingPairs / self.verifiedPairs))
            else:
              print('array is empty 3')
            print("Verified pairs", str(self.verifiedPairs))


        def verifyRelations(self, geomIds1, geomIds2, sourceGeoms, targetGeoms):
            import numpy as np
            from shapely.wkb import dumps as wkb_dumps

            N = len(sourceGeoms)

            self.fast_geom = FastGeom()

            # --- 2. Call fast C++ relate ---
            res_ptr = self.fast_geom.relate(sourceGeoms, targetGeoms)

            # --- 3. Decode DE-9IM from uint64_t safely ---
            def decode_de9im_safe(u64):
                u = int(u64)
                if u == 0:
                    return 'FFFFFFFFF'  # disjoint fallback
                return ''.join(chr((u >> (8 * i)) & 0xFF) for i in range(8)) + 'F'

            de9im_str_array = [decode_de9im_safe(res_ptr[i]) for i in range(N)]

            # Optional: track errors
            num_invalid = sum(1 for val in res_ptr[:N] if val == 0)
            if num_invalid > 0:
                print(f"  Skipped {num_invalid} invalid geometries (GEOSRelate returned 0)")

            # Free memory
            #self.lib.free_result_u64(res_ptr)

            # --- 4. Continue with vectorized logic ---
            self.verifiedPairs += N
            encoded = encode_de9im_strings(de9im_str_array)
            dimensions1 = self.geom_utils.get_dimensions(sourceGeoms)
            dimensions2 = self.geom_utils.get_dimensions(targetGeoms)

            same_dim = (dimensions1 == dimensions2)
            both_dim1 = (dimensions1 == 1) & (dimensions2 == 1)
            d1_greater_d2 = (dimensions1 > dimensions2)
            d1_less_d2 = (dimensions1 < dimensions2)

            # --- 5. Match relationships ---
            mask_intersects    = intersects_de9im.matches_array(encoded)
            mask_within        = within.matches_array(encoded)
            mask_covered_by    = covered_by.matches_array(encoded)
            mask_equal         = equal.matches_array(encoded)
            mask_touches       = touches.matches_array(encoded)
            mask_contains      = contains.matches_array(encoded)
            mask_covers        = covers.matches_array(encoded)
            mask_overlaps1     = overlaps1.matches_array(encoded)
            mask_overlaps2     = overlaps2.matches_array(encoded)
            mask_overlaps      = same_dim & (mask_overlaps1 | mask_overlaps2)
            mask_crosses_lines = both_dim1 & crosses_lines.matches_array(encoded)
            mask_crosses_1     = d1_less_d2 & crosses_1.matches_array(encoded)
            mask_crosses_2     = d1_greater_d2 & crosses_2.matches_array(encoded)

            # --- 6. Apply relationships ---
            def process_mask(mask, add_method):
                idx = np.where(mask)[0]
                for i in idx:
                    add_method(geomIds1[i], geomIds2[i])
                return len(idx)

            self.detectedLinks += process_mask(mask_intersects, self.addIntersects)
            self.detectedLinks += process_mask(mask_within, self.addWithin)
            self.detectedLinks += process_mask(mask_covered_by, self.addCoveredBy)
            self.detectedLinks += process_mask(mask_overlaps, self.addOverlaps)
            self.detectedLinks += process_mask(mask_crosses_lines, self.addCrosses)
            self.detectedLinks += process_mask(mask_crosses_1, self.addCrosses)
            self.detectedLinks += process_mask(mask_crosses_2, self.addCrosses)
            self.detectedLinks += process_mask(mask_equal, self.addEquals)
            self.detectedLinks += process_mask(mask_touches, self.addTouches)
            self.detectedLinks += process_mask(mask_contains, self.addContains)
            self.detectedLinks += process_mask(mask_covers, self.addCovers)

            # --- 7. Final related mask ---
            related_mask = (
                mask_intersects | mask_within | mask_covered_by |
                mask_equal | mask_touches | mask_contains | mask_covers |
                mask_overlaps | mask_crosses_lines | mask_crosses_1 | mask_crosses_2
            )

            n_related = np.sum(related_mask)
            self.interlinkedGeometries += n_related
            self.pgr += n_related

            for is_related in related_mask:
                if is_related:
                    self.continuous_unrelated_Pairs = 0
                else:
                    self.continuous_unrelated_Pairs += 1

            return related_mask


