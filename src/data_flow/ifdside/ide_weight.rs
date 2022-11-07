use super::super::JoinLattice;

pub trait IDEWeight<L>: JoinLattice {
    fn compose(first: Self, second: Self) -> Self;
    fn compute(&self, source: L) -> L;
}
