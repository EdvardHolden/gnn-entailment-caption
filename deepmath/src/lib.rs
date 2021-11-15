use numpy::{PyArray, PyArray1};
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Directed;
use pyo3::prelude::*;
use std::collections::{BTreeSet, HashMap};
use tptp::common::*;
use tptp::fof;
use tptp::top::FofAnnotated;
use tptp::visitor::Visitor;
use tptp::Parse;

#[derive(Debug, Clone, Copy)]
enum NodeType {
    True,
    False,
    Variable,
    Functor,
    Argument,
    Application,
    Equality,
    Negation,
    And,
    Or,
    Equivalent,
    Forall,
    Exists,
    Axiom,
    Conjecture,
}

#[derive(Default)]
struct GraphBuilder {
    graph: Graph<NodeType, (), Directed, u32>,
    last: NodeIndex,
    functors: HashMap<String, NodeIndex>,
    variables: HashMap<String, NodeIndex>,
    terms: HashMap<Vec<NodeIndex>, NodeIndex>,
    equalities: HashMap<(NodeIndex, NodeIndex), NodeIndex>,
    negations: HashMap<NodeIndex, NodeIndex>,
    conjunctions: HashMap<BTreeSet<NodeIndex>, NodeIndex>,
    disjunctions: HashMap<BTreeSet<NodeIndex>, NodeIndex>,
}

impl GraphBuilder {
    fn equality(&mut self, left: &fof::Term, right: &fof::Term) {
        self.visit_fof_term(left);
        let left = self.last;
        self.visit_fof_term(right);
        let right = self.last;

        self.last = if let Some(node) = self.equalities.get(&(left, right)) {
            *node
        } else {
            let equality = self.graph.add_node(NodeType::Equality);
            self.graph.add_edge(equality, left, ());
            self.graph.add_edge(equality, right, ());
            self.equalities.insert((left, right), equality);
            self.equalities.insert((right, left), equality);
            equality
        }
    }

    fn visit(&mut self, fof: FofAnnotated, conjecture: bool) -> NodeIndex {
        self.variables.clear();
        self.visit_fof_formula(&fof.0.formula);
        let node_type = if conjecture {
            NodeType::Conjecture
        } else {
            NodeType::Axiom
        };
        let formula = self.last;
        let marker = self.graph.add_node(node_type);
        self.graph.add_edge(marker, formula, ());
        marker
    }

    fn finish(self) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        let nodes = self
            .graph
            .raw_nodes()
            .iter()
            .map(|n| n.weight as i64)
            .collect();
        let (sources, targets) = self
            .graph
            .raw_edges()
            .iter()
            .map(|e| (e.source().index() as i64, e.target().index() as i64))
            .unzip();

        (nodes, sources, targets)
    }
}

impl<'v> Visitor<'v> for GraphBuilder {
    fn visit_variable(&mut self, variable: &Variable) {
        let key = format!("{}", variable);
        let variables = &mut self.variables;
        let graph = &mut self.graph;
        self.last = *variables
            .entry(key)
            .or_insert_with(|| graph.add_node(NodeType::Variable));
    }

    fn visit_functor(&mut self, functor: &Functor) {
        let key = format!("{}", functor);
        let functors = &mut self.functors;
        let graph = &mut self.graph;
        self.last = *functors
            .entry(key)
            .or_insert_with(|| graph.add_node(NodeType::Functor));
    }

    fn visit_defined_term(&mut self, defined: &DefinedTerm) {
        let key = format!("{}", defined);
        let functors = &mut self.functors;
        let graph = &mut self.graph;
        self.last = *functors
            .entry(key)
            .or_insert_with(|| graph.add_node(NodeType::Functor));
    }

    fn visit_fof_defined_plain_formula(&mut self, defined: &fof::DefinedPlainFormula) {
        let value = format!("{}", defined);
        let node_type = if value == "$true" {
            NodeType::True
        } else if value == "$false" {
            NodeType::False
        } else {
            unimplemented!()
        };
        self.last = self.graph.add_node(node_type);
    }

    fn visit_fof_plain_term(&mut self, term: &fof::PlainTerm) {
        use fof::PlainTerm::*;
        let (functor, arguments) = match term {
            Constant(constant) => (&constant.0, &[] as &[fof::Term]),
            Function(functor, arguments) => (functor, arguments.0.as_slice()),
        };
        self.visit_functor(functor);
        let functor = self.last;

        let mut children = vec![];
        for argument in arguments {
            self.visit_fof_term(argument);
            children.push(self.last);
        }

        let mut key = vec![functor];
        key.extend_from_slice(&children);
        let terms = &mut self.terms;
        let graph = &mut self.graph;
        self.last = *terms.entry(key).or_insert_with(|| {
            let mut arg_nodes = vec![];
            for argument in children {
                let arg_node = graph.add_node(NodeType::Argument);
                graph.add_edge(arg_node, argument, ());
                arg_nodes.push(arg_node);
            }
            let app_node = graph.add_node(NodeType::Application);
            graph.add_edge(app_node, functor, ());
            for arg_node in &arg_nodes {
                graph.add_edge(app_node, *arg_node, ());
            }
            for i in 1..arg_nodes.len() {
                graph.add_edge(arg_nodes[i - 1], arg_nodes[i], ());
            }
            app_node
        });
    }

    fn visit_fof_defined_infix_formula(
        &mut self,
        defined_infix_formula: &fof::DefinedInfixFormula,
    ) {
        self.equality(&defined_infix_formula.left, &defined_infix_formula.right);
    }

    fn visit_fof_unary_formula(&mut self, unary_formula: &fof::UnaryFormula) {
        match unary_formula {
            fof::UnaryFormula::Unary(_, formula) => {
                self.visit_fof_unit_formula(formula);
            }
            fof::UnaryFormula::InfixUnary(formula) => {
                self.equality(&formula.left, &formula.right);
            }
        };
        let formula = self.last;

        self.last = if let Some(node) = self.negations.get(&formula) {
            *node
        } else {
            let negation = self.graph.add_node(NodeType::Negation);
            self.graph.add_edge(negation, formula, ());
            negation
        }
    }

    fn visit_fof_and_formula(&mut self, and_formula: &fof::AndFormula) {
        let mut children = BTreeSet::new();
        for formula in &and_formula.0 {
            self.visit_fof_unit_formula(formula);
            children.insert(self.last);
        }

        self.last = if let Some(node) = self.conjunctions.get(&children) {
            *node
        } else {
            let conjunction = self.graph.add_node(NodeType::And);
            for child in children {
                self.graph.add_edge(conjunction, child, ());
            }
            conjunction
        }
    }

    fn visit_fof_or_formula(&mut self, or_formula: &fof::OrFormula) {
        let mut children = BTreeSet::new();
        for formula in &or_formula.0 {
            self.visit_fof_unit_formula(formula);
            children.insert(self.last);
        }

        self.last = if let Some(node) = self.disjunctions.get(&children) {
            *node
        } else {
            let disjunction = self.graph.add_node(NodeType::Or);
            for child in children {
                self.graph.add_edge(disjunction, child, ());
            }
            disjunction
        }
    }

    fn visit_fof_binary_nonassoc(&mut self, nonassoc: &fof::BinaryNonassoc) {
        self.visit_fof_unit_formula(&nonassoc.left);
        let left = self.last;
        self.visit_fof_unit_formula(&nonassoc.right);
        let right = self.last;

        self.last = match nonassoc.op {
            NonassocConnective::Equivalent => {
                let equiv = self.graph.add_node(NodeType::Equivalent);
                self.graph.add_edge(equiv, left, ());
                self.graph.add_edge(equiv, right, ());
                equiv
            }
            NonassocConnective::LRImplies => {
                let marker = self.graph.add_node(NodeType::Negation);
                let implies = self.graph.add_node(NodeType::Or);
                self.graph.add_edge(marker, left, ());
                self.graph.add_edge(implies, marker, ());
                self.graph.add_edge(implies, right, ());
                implies
            }
            _ => unimplemented!(),
        }
    }

    fn visit_fof_quantified_formula(&mut self, quantified: &fof::QuantifiedFormula) {
        let node_type = if quantified.quantifier == fof::Quantifier::Forall {
            NodeType::Forall
        } else {
            NodeType::Exists
        };
        let node = self.graph.add_node(node_type);
        for variable in &quantified.bound.0 {
            self.visit_variable(variable);
            let variable = self.last;
            self.graph.add_edge(node, variable, ());
        }
        self.visit_fof_unit_formula(&quantified.formula);
        let formula = self.last;
        self.graph.add_edge(node, formula, ());
        self.last = node;
    }
}

#[pymodule]
fn parser(_py: Python, module: &PyModule) -> PyResult<()> {
    type LongTensor = PyArray1<i64>;

    #[pyfn(module)]
    fn graph<'p>(
        py: Python<'p>,
        conjectures: Vec<&[u8]>,
        premises: Vec<&[u8]>,
    ) -> PyResult<(
        &'p LongTensor,
        &'p LongTensor,
        &'p LongTensor,
        &'p LongTensor,
        &'p LongTensor,
    )> {

        let mut premise_indices = vec![];
        let mut conjecture_indices = vec![];
        let mut builder = GraphBuilder::default();

        for conjecture in conjectures {
            let (_, conjecture) =
                <FofAnnotated as Parse<'_, ()>>::parse(conjecture).expect("parse error");
            conjecture_indices.push(builder.visit(conjecture, true).index() as i64);
        }

        for premise in premises {
            let (_, premise) =
                <FofAnnotated as Parse<'_, ()>>::parse(premise).expect("parse error");
            premise_indices.push(builder.visit(premise, false).index() as i64);
        }

        let (nodes, sources, targets) = builder.finish();
        let nodes = PyArray::from_vec(py, nodes);
        let sources = PyArray::from_vec(py, sources);
        let targets = PyArray::from_vec(py, targets);
        let premise_indices = PyArray::from_vec(py, premise_indices);
        let conjecture_indices = PyArray::from_vec(py, conjecture_indices);
        Ok((nodes, sources, targets, premise_indices, conjecture_indices))
    }

    Ok(())
}
