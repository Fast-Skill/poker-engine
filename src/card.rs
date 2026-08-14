//! A card is a `u8` in `0..52`, encoded as `rank * 4 + suit`.
//!
//! Ranks run 0 (deuce) through 12 (ace); suits run 0..4 as clubs, diamonds,
//! hearts, spades. Packing rank into the high bits means `card >> 2` is the
//! rank and `card & 3` is the suit, both single instructions.

pub const RANK_CHARS: [u8; 13] = *b"23456789TJQKA";
pub const SUIT_CHARS: [u8; 4] = *b"cdhs";

/// Sentinel for "no card". Never a legal index.
pub const NO_CARD: u8 = 52;

#[inline]
pub fn card(rank: u8, suit: u8) -> u8 {
    debug_assert!(rank < 13 && suit < 4);
    rank * 4 + suit
}

#[inline]
pub fn rank_of(c: u8) -> u8 {
    c >> 2
}

#[inline]
pub fn suit_of(c: u8) -> u8 {
    c & 3
}

pub fn to_string(c: u8) -> String {
    if c >= 52 {
        return "??".to_string();
    }
    let r = RANK_CHARS[rank_of(c) as usize] as char;
    let s = SUIT_CHARS[suit_of(c) as usize] as char;
    format!("{r}{s}")
}

/// Parse a single card such as `"As"` or `"Td"`.
pub fn parse(s: &str) -> Option<u8> {
    let b = s.as_bytes();
    if b.len() != 2 {
        return None;
    }
    let rank = RANK_CHARS.iter().position(|&r| r == b[0].to_ascii_uppercase())?;
    let suit = SUIT_CHARS.iter().position(|&x| x == b[1].to_ascii_lowercase())?;
    Some(card(rank as u8, suit as u8))
}

/// Parse a run of cards such as `"AsKsQsJsTs"`, with optional whitespace.
pub fn parse_many(s: &str) -> Option<Vec<u8>> {
    let cleaned: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if !cleaned.len().is_multiple_of(2) {
        return None;
    }
    cleaned
        .as_bytes()
        .chunks(2)
        .map(|c| parse(std::str::from_utf8(c).ok()?))
        .collect()
}

pub fn cards_to_string(cards: &[u8]) -> String {
    cards.iter().map(|&c| to_string(c)).collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_every_card() {
        for c in 0..52u8 {
            assert_eq!(parse(&to_string(c)), Some(c));
        }
    }

    #[test]
    fn known_encodings() {
        assert_eq!(parse("2c"), Some(0));
        assert_eq!(parse("As"), Some(51));
        assert_eq!(rank_of(parse("Td").unwrap()), 8);
        assert_eq!(suit_of(parse("Td").unwrap()), 1);
    }

    #[test]
    fn rejects_garbage() {
        assert_eq!(parse("Xs"), None);
        assert_eq!(parse("Az"), None);
        assert_eq!(parse("A"), None);
        assert_eq!(parse_many("AsK"), None);
    }

    #[test]
    fn parses_a_board() {
        let b = parse_many("Ah Kd 7c 7d 2s").unwrap();
        assert_eq!(b.len(), 5);
        assert_eq!(cards_to_string(&b), "Ah Kd 7c 7d 2s");
    }
}
