"""Unit tests for app/rag/hierarchy.py.

These specifically lock in the two real bugs found once the doc_type
slug set was confirmed against `tmquan/vbpl-vn`'s own build pipeline
(codes.py) rather than the dataset card's top-12 table alone — see
hierarchy.py's module docstring for the full explanation:

  1. There is no "nghi_quyet_qh" slug in the real data — the single
     slug "nghi_quyet" is used for both a National Assembly resolution
     and a provincial HĐND one. Before the fix, EVERY nghị quyết
     (16.4% of the corpus) fell to UNRANKED_FALLBACK regardless of
     level.
  2. "sac_lenh" (0.6% of the corpus, a real normative instrument) was
     entirely absent from the rank map for the same reason.
"""
from __future__ import annotations

from app.rag.hierarchy import UNRANKED_FALLBACK, is_ranked, more_authoritative, rank_of


class TestNghiQuyetScopeAwareRanking:
    def test_central_nghi_quyet_ranks_at_the_luat_adjacent_tier(self):
        assert rank_of("nghi_quyet", "trung_uong") == rank_of("luat")

    def test_provincial_nghi_quyet_ranks_at_the_implementing_tier(self):
        assert rank_of("nghi_quyet", "dia_phuong") == rank_of("quyet_dinh")

    def test_central_nghi_quyet_outranks_provincial_nghi_quyet(self):
        assert rank_of("nghi_quyet", "trung_uong") < rank_of("nghi_quyet", "dia_phuong")

    def test_nghi_quyet_without_scope_still_gets_a_real_tier_not_the_fallback(self):
        # The old bug: doc_type "nghi_quyet" alone (no scope info) fell
        # all the way to UNRANKED_FALLBACK because only the fictional
        # "nghi_quyet_qh" slug was in the map. Never true in the real
        # data, so this must resolve to a real tier now.
        assert rank_of("nghi_quyet") != UNRANKED_FALLBACK
        assert rank_of("nghi_quyet") == rank_of("quyet_dinh")

    def test_nonexistent_slug_nghi_quyet_qh_is_not_special_cased(self):
        # Confirms the fictional slug from the old map is gone — an
        # actual row would never carry this value, so it must fall
        # through to the generic unranked fallback like any other
        # unrecognised doc_type, not be treated as meaningful.
        assert rank_of("nghi_quyet_qh") == UNRANKED_FALLBACK

    def test_is_ranked_is_true_for_nghi_quyet_regardless_of_scope(self):
        assert is_ranked("nghi_quyet") is True
        assert is_ranked("nghi_quyet", "trung_uong") is True
        assert is_ranked("nghi_quyet", "dia_phuong") is True


class TestPreviouslyMissingNormativeSlugs:
    def test_sac_lenh_is_ranked_not_fallback(self):
        assert is_ranked("sac_lenh") is True
        assert rank_of("sac_lenh") != UNRANKED_FALLBACK
        # Same real-world tier as lệnh (a Head-of-State instrument).
        assert rank_of("sac_lenh") == rank_of("lenh")

    def test_sac_luat_ranks_at_the_luat_tier(self):
        assert rank_of("sac_luat") == rank_of("luat")

    def test_nghi_quyet_lien_tich_is_ranked(self):
        assert is_ranked("nghi_quyet_lien_tich") is True

    def test_van_ban_hop_nhat_is_deliberately_unranked(self):
        # Its own slug carries no signal about which original
        # instrument it consolidates — see hierarchy.py's note.
        assert is_ranked("van_ban_hop_nhat") is False
        assert rank_of("van_ban_hop_nhat") == UNRANKED_FALLBACK


class TestMoreAuthoritative:
    def test_luat_outranks_quyet_dinh(self):
        assert more_authoritative("luat", "quyet_dinh") is True
        assert more_authoritative("quyet_dinh", "luat") is False

    def test_equal_tier_is_not_authoritative_either_direction(self):
        assert more_authoritative("quyet_dinh", "chi_thi") is False
        assert more_authoritative("chi_thi", "quyet_dinh") is False

    def test_scope_aware_for_nghi_quyet_on_both_sides(self):
        assert (
            more_authoritative(
                "nghi_quyet", "nghi_quyet", scope_a="trung_uong", scope_b="dia_phuong"
            )
            is True
        )
