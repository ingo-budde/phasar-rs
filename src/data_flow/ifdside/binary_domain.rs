use crate::data_flow::JoinLattice;

#[derive(PartialEq)]
pub enum BinaryDomain {
    Top,
    Bottom,
}

impl JoinLattice for BinaryDomain {
    fn join(lhs: Self, rhs: Self) -> Self {
        if lhs.is_top() || rhs.is_top() {
            Self::Top
        } else {
            Self::Bottom
        }
    }

    fn top_value() -> Self {
        Self::Top
    }

    fn is_top(&self) -> bool {
        match self {
            BinaryDomain::Top => true,
            BinaryDomain::Bottom => false,
        }
    }
}
