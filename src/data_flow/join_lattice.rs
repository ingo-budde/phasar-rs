pub trait JoinLattice: PartialEq {
    fn join(lhs: Self, rhs: Self) -> Self;
    fn top_value() -> Self;
    fn is_top(&self) -> bool;
}
