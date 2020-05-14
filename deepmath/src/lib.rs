use numpy::{PyArray, PyArray1};
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Directed;
use pyo3::prelude::*;
use std::collections::HashMap;
use tptp::parsers::fof_annotated;
use tptp::syntax::*;
use tptp::visitor::Visitor;

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
    Implication,
    Equivalent,
    Forall,
    Exists,
    Axiom,
    Conjecture,
}

#[derive(Default)]
struct GraphBuilder {
    graph: Graph<NodeType, (), Directed, u32>,
    stack: Vec<Vec<NodeIndex>>,
    functors: HashMap<String, NodeIndex>,
    variables: HashMap<String, NodeIndex>,
    terms: HashMap<Vec<NodeIndex>, NodeIndex>,
}

impl GraphBuilder {
    fn children(&mut self) -> Vec<NodeIndex> {
        self.stack.pop().unwrap()
    }

    fn level(&mut self) {
        self.stack.push(vec![]);
    }

    fn last(&mut self) -> NodeIndex {
        self.stack.last_mut().unwrap().pop().unwrap()
    }

    fn record(&mut self, child: NodeIndex) {
        self.stack.last_mut().unwrap().push(child);
    }

    fn visit(&mut self, fof: FofAnnotated, conjecture: bool) {
        self.variables.clear();
        self.stack.clear();
        self.level();
        self.visit_fof_formula(fof.formula);
        let node_type = if conjecture {
            NodeType::Conjecture
        } else {
            NodeType::Axiom
        };
        let formula = self.last();
        let marker = self.graph.add_node(node_type);
        self.graph.add_edge(marker, formula, ());
    }

    fn finish(self) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        /*
        for node_index in self.graph.node_indices() {
            self.graph.add_edge(node_index, node_index, ());
        }
        */

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
    fn visit_variable(&mut self, variable: Variable) {
        let key = format!("{}", variable);
        let variables = &mut self.variables;
        let graph = &mut self.graph;
        let index = *variables
            .entry(key)
            .or_insert_with(|| graph.add_node(NodeType::Variable));
        self.record(index)
    }

    fn visit_functor(&mut self, functor: Functor) {
        let key = format!("{}", functor);
        let functors = &mut self.functors;
        let graph = &mut self.graph;
        let index = *functors
            .entry(key)
            .or_insert_with(|| graph.add_node(NodeType::Functor));
        self.record(index)
    }

    fn visit_defined_term(&mut self, defined: DefinedTerm) {
        let key = format!("{}", defined);
        let functors = &mut self.functors;
        let graph = &mut self.graph;
        let index = *functors
            .entry(key)
            .or_insert_with(|| graph.add_node(NodeType::Functor));
        self.record(index)
    }

    fn visit_fof_defined_plain_formula(&mut self, defined: FofDefinedPlainFormula) {
        let value = format!("{}", defined);
        let node_type = if value == "$true" {
            NodeType::True
        } else if value == "$false" {
            NodeType::False
        } else {
            unimplemented!()
        };
        let node = self.graph.add_node(node_type);
        self.record(node);
    }

    fn visit_fof_plain_term(&mut self, term: FofPlainTerm) {
        use FofPlainTerm::*;
        match term {
            Constant(constant) => self.visit_constant(constant),
            Function(functor, arguments) => {
                self.level();
                self.visit_functor(functor);
                for argument in arguments.0 {
                    self.visit_fof_term(argument);
                }
                let key = self.children();

                let terms = &mut self.terms;
                let graph = &mut self.graph;
                let index = *terms.entry(key.clone()).or_insert_with(|| {
                    let mut arg_nodes = vec![];
                    for argument in &key[1..] {
                        let arg_node = graph.add_node(NodeType::Argument);
                        graph.add_edge(arg_node, *argument, ());
                        arg_nodes.push(arg_node);
                    }
                    let app_node = graph.add_node(NodeType::Application);
                    graph.add_edge(app_node, key[0], ());
                    for arg_node in &arg_nodes {
                        graph.add_edge(app_node, *arg_node, ());
                    }
                    for i in 1..arg_nodes.len() {
                        graph.add_edge(arg_nodes[i - 1], arg_nodes[i], ());
                    }
                    app_node
                });
                self.record(index)
            }
        }
    }

    fn visit_fof_defined_infix_formula(&mut self, defined_infix_formula: FofDefinedInfixFormula) {
        self.visit_fof_term(defined_infix_formula.left);
        self.visit_fof_term(defined_infix_formula.right);
        let right = self.last();
        let left = self.last();
        let equality = self.graph.add_node(NodeType::Equality);
        self.graph.add_edge(equality, left, ());
        self.graph.add_edge(equality, right, ());
        self.record(equality);
    }

    fn visit_fof_unary_formula(&mut self, unary_formula: FofUnaryFormula) {
        match unary_formula {
            FofUnaryFormula::Unary(_, formula) => {
                let not = self.graph.add_node(NodeType::Negation);
                self.visit_fof_unit_formula(formula);
                let formula = self.last();
                self.graph.add_edge(not, formula, ());
                self.record(not);
            }
            _ => unimplemented!(),
        }
    }

    fn visit_fof_or_formula(&mut self, or_formula: FofOrFormula) {
        let or = self.graph.add_node(NodeType::Or);
        for formula in or_formula.0 {
            self.visit_fof_unit_formula(formula);
            let formula = self.last();
            self.graph.add_edge(or, formula, ());
        }
        self.record(or);
    }

    fn visit_fof_and_formula(&mut self, and_formula: FofAndFormula) {
        let and = self.graph.add_node(NodeType::And);
        for formula in and_formula.0 {
            self.visit_fof_unit_formula(formula);
            let formula = self.last();
            self.graph.add_edge(and, formula, ());
        }
        self.record(and);
    }

    fn visit_fof_binary_nonassoc(&mut self, nonassoc: FofBinaryNonassoc) {
        self.visit_fof_unit_formula(nonassoc.left);
        let left = self.last();
        self.visit_fof_unit_formula(nonassoc.right);
        let right = self.last();
        match nonassoc.op {
            NonassocConnective::Equivalent => {
                let equiv = self.graph.add_node(NodeType::Equivalent);
                self.graph.add_edge(equiv, left, ());
                self.graph.add_edge(equiv, right, ());
                self.record(equiv);
            }
            NonassocConnective::LRImplies => {
                let implies = self.graph.add_node(NodeType::Implication);
                self.graph.add_edge(left, right, ());
                self.graph.add_edge(implies, left, ());
                self.graph.add_edge(implies, right, ());
                self.record(implies);
            }
            _ => unimplemented!(),
        }
    }

    fn visit_fof_quantified_formula(&mut self, quantified: FofQuantifiedFormula) {
        let node_type = if quantified.quantifier == FofQuantifier::Forall {
            NodeType::Forall
        } else {
            NodeType::Exists
        };
        let node = self.graph.add_node(node_type);
        for variable in quantified.bound.0 {
            self.visit_variable(variable);
            let variable = self.last();
            self.graph.add_edge(node, variable, ());
        }
        self.visit_fof_unit_formula(quantified.formula);
        let formula = self.last();
        self.graph.add_edge(node, formula, ());
        self.record(node);
    }
}

#[pymodule]
fn parser(_py: Python, module: &PyModule) -> PyResult<()> {
    type LongTensor = PyArray1<i64>;

    #[pyfn(module, "graph")]
    fn to_graph<'p>(
        py: Python<'p>,
        conjecture: &[u8],
        axiom: &[u8],
    ) -> PyResult<(&'p LongTensor, &'p LongTensor, &'p LongTensor)> {
        let mut builder = GraphBuilder::default();
        builder.level();

        let (_, axiom) = fof_annotated::<()>(axiom).expect("parse error");
        builder.visit(axiom, false);
        let (_, conjecture) = fof_annotated::<()>(conjecture).expect("parse error");
        builder.visit(conjecture, true);

        let (nodes, sources, targets) = builder.finish();
        let nodes = PyArray::from_vec(py, nodes);
        let sources = PyArray::from_vec(py, sources);
        let targets = PyArray::from_vec(py, targets);
        Ok((nodes, sources, targets))
    }

    Ok(())
}
