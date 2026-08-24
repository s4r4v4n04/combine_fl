"""
    ======================================================
    Python bindings for Universal Key Value store library.
    ======================================================
    
    Supports:
    **********
    * Collection-level CRUD operations, like `dict`.
    * Batch operations & ACID transactions.
    * Graph collections, mimicking `networkx`.
    * Tabular views, mimicking `pandas`.
    * Apache Arrow exports for inter-process communication.
    
    """
from __future__ import annotations
import ukv.leveldb
import typing
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "Collection",
    "DataBase",
    "DataFrame",
    "DegreeView",
    "DegreesStream",
    "DocsCollection",
    "DocsKVRange",
    "DocsKVStream",
    "EdgesIter",
    "EdgesRange",
    "EdgesStream",
    "ItemsRange",
    "ItemsStream",
    "KeysRange",
    "KeysStream",
    "Network",
    "NodesRange",
    "NodesStream",
    "Transaction",
    "allows_loops",
    "density",
    "from_dict",
    "from_records",
    "is_directed",
    "is_multi",
    "write_adjlist"
]


class Collection():
    def __contains__(self, arg0: object) -> object: ...
    def __delitem__(self, arg0: object) -> None: ...
    def __getitem__(self, arg0: object) -> object: ...
    def __len__(self) -> int: ...
    def __setitem__(self, arg0: object, arg1: object) -> None: ...
    def broadcast(self, arg0: object, arg1: object) -> None: ...
    def clear(self) -> None: ...
    def get(self, arg0: object) -> object: ...
    def get_matrix(self, arg0: object, arg1: int, arg2: str) -> int: ...
    def has_key(self, arg0: object) -> object: ...
    def pop(self, arg0: object) -> None: ...
    def remove(self) -> None: ...
    def sample_keys(self, arg0: int) -> object: ...
    def scan(self, arg0: int, arg1: int) -> numpy.ndarray[numpy.int64]: ...
    def set(self, arg0: object, arg1: object) -> None: ...
    def set_matrix(self, arg0: object, arg1: object) -> int: ...
    def update(self, arg0: object) -> None: ...
    @property
    def docs(self) -> unum::ukv::py_collection_gt<unum::ukv::docs_collection_t>:
        """
        :type: unum::ukv::py_collection_gt<unum::ukv::docs_collection_t>
        """
    @property
    def graph(self) -> object:
        """
        :type: object
        """
    @property
    def items(self) -> object:
        """
        :type: object
        """
    @property
    def keys(self) -> object:
        """
        :type: object
        """
    @property
    def media(self) -> int:
        """
        :type: int
        """
    @property
    def table(self) -> object:
        """
        :type: object
        """
    pass
class DataBase():
    def __contains__(self, collection: str) -> bool: ...
    def __delitem__(self, collection: str) -> None: ...
    def __enter__(self) -> DataBase: ...
    def __exit__(self, arg0: object, arg1: object, arg2: object) -> bool: ...
    def __getitem__(self, collection: str) -> Collection: ...
    def __init__(self, config: str = '', open: bool = True, prefer_arrow: bool = True) -> None: ...
    def clear(self) -> None: ...
    def close(self) -> None: ...
    def collection_names(self) -> typing.List[str]: ...
    @property
    def main(self) -> Collection:
        """
        :type: Collection
        """
    pass
class DataFrame():
    def __getitem__(self, arg0: handle) -> DataFrame: ...
    def __init__(self, db: DataBase, index: str = '') -> None: ...
    def astype(self, arg0: handle) -> DataFrame: ...
    def drop(self, arg0: object) -> None: ...
    def head(self, arg0: int) -> DataFrame: ...
    @typing.overload
    def insert(self, arg0: object) -> None: ...
    @typing.overload
    def insert(self, arg0: str, arg1: object) -> None: ...
    def loc(self, arg0: handle) -> DataFrame: ...
    def merge(self, arg0: DataFrame) -> None: ...
    def rename(self, arg0: object) -> None: ...
    def sample(self, arg0: int) -> DataFrame: ...
    def tail(self, arg0: int) -> DataFrame: ...
    def to_arrow(self) -> object: ...
    def to_csv(self, arg0: str) -> None: ...
    def to_json(self, path: str = '') -> object: ...
    def to_parquet(self, arg0: str) -> None: ...
    def update(self, arg0: object) -> None: ...
    @property
    def empty(self) -> bool:
        """
        :type: bool
        """
    @property
    def shape(self) -> tuple:
        """
        :type: tuple
        """
    @property
    def size(self) -> int:
        """
        :type: int
        """
    pass
class DegreeView():
    @typing.overload
    def __call__(self, vs: object, weight: str = '') -> object: ...
    @typing.overload
    def __call__(self, weight: str = '') -> DegreeView: ...
    def __getitem__(self, arg0: int) -> int: ...
    def __iter__(self) -> DegreesStream: ...
    pass
class DegreesStream():
    def __next__(self) -> object: ...
    pass
class DocsCollection():
    def __contains__(self, arg0: object) -> object: ...
    def __delitem__(self, arg0: object) -> None: ...
    def __getitem__(self, arg0: object) -> object: ...
    def __len__(self) -> int: ...
    def __setitem__(self, arg0: object, arg1: object) -> None: ...
    def broadcast(self, arg0: object, arg1: object) -> None: ...
    def clear(self) -> None: ...
    def get(self, arg0: object) -> object: ...
    def has_key(self, arg0: object) -> object: ...
    def merge(self, arg0: object, arg1: object) -> None: ...
    def patch(self, arg0: object, arg1: object) -> None: ...
    def remove(self, arg0: object) -> None: ...
    def scan(self, arg0: int, arg1: int) -> object: ...
    def set(self, arg0: object, arg1: object) -> None: ...
    @property
    def items(self) -> object:
        """
        :type: object
        """
    @property
    def keys(self) -> object:
        """
        :type: object
        """
    pass
class DocsKVRange():
    def __iter__(self) -> DocsKVStream: ...
    def since(self, arg0: int) -> DocsKVRange: ...
    def until(self, arg0: int) -> DocsKVRange: ...
    pass
class DocsKVStream():
    def __next__(self) -> tuple: ...
    pass
class EdgesIter():
    def __next__(self) -> object: ...
    pass
class EdgesRange():
    @typing.overload
    def __call__(self, data: bool = False) -> EdgesRange: ...
    @typing.overload
    def __call__(self, data: str, default: object = None) -> EdgesRange: ...
    @typing.overload
    def __call__(self, vs: object, data: bool) -> EdgesRange: ...
    @typing.overload
    def __call__(self, vs: object, data: str, default: object = None) -> EdgesRange: ...
    def __iter__(self) -> object: ...
    pass
class EdgesStream():
    def __next__(self) -> object: ...
    pass
class ItemsRange():
    def __iter__(self) -> ItemsStream: ...
    def since(self, arg0: int) -> ItemsRange: ...
    def until(self, arg0: int) -> ItemsRange: ...
    pass
class ItemsStream():
    def __next__(self) -> tuple: ...
    pass
class KeysRange():
    def __iter__(self) -> KeysStream: ...
    def since(self, arg0: int) -> KeysRange: ...
    def until(self, arg0: int) -> KeysRange: ...
    pass
class KeysStream():
    def __next__(self) -> int: ...
    pass
class Network():
    def __contains__(self, n: int) -> bool: 
        """
        Returns True if the graph contains the node n.
        """
    def __getitem__(self, n: int) -> object: 
        """
        Returns an iterable of incoming and outgoing nodes of n. Potentially with duplicates.
        """
    def __init__(self, db: DataBase, index: typing.Optional[str] = None, vertices: typing.Optional[str] = None, relations: typing.Optional[str] = None, directed: bool = False, multi: bool = False, loops: bool = False) -> None: ...
    def __iter__(self) -> None: 
        """
        Iterate over the nodes.
        """
    def __len__(self) -> int: 
        """
        Returns the number of nodes in the graph.
        """
    @typing.overload
    def add_edge(self, u_for_edge: int, v_for_edge: int) -> None: ...
    @typing.overload
    def add_edge(self, u_for_edge: int, v_for_edge: int, id: int, **kwargs) -> None: ...
    @typing.overload
    def add_edges_from(self, ebunch_to_add: object) -> None: 
        """
        Adds an adjacency list (in a form of 2 or 3 columnar matrix) to the graph.

        Adds edges from members of the first array to members of the second array.
        """
    @typing.overload
    def add_edges_from(self, us: object, vs: object, keys: object = None, **kwargs) -> None: ...
    def add_node(self, v_to_upsert: int, **kwargs) -> None: ...
    def add_nodes_from(self, arg0: object, **kwargs) -> None: ...
    def clear(self) -> None: 
        """
        Removes both vertices and edges from the graph.
        """
    def clear_edges(self) -> None: 
        """
        Removes edges from the graph.
        """
    def community_louvain(self) -> object: 
        """
        Community Louvain.
        """
    def copy(self) -> None: ...
    def edge_subgraph(self) -> None: ...
    def get_edge_attributes(self, name: str) -> object: ...
    def get_edge_data(self, u: int, v: int) -> None: ...
    def get_node_attributes(self, name: str) -> object: ...
    @typing.overload
    def has_edge(self, u: int, v: int) -> bool: ...
    @typing.overload
    def has_edge(self, u: int, v: int, key: int) -> bool: ...
    def has_node(self, n: int) -> bool: 
        """
        Returns True if the graph contains the node n.
        """
    def is_directed(self) -> bool: ...
    def is_multigraph(self) -> bool: ...
    def nbunch_iter(self, arg0: _object) -> numpy.ndarray[numpy.int64]: 
        """
        Filters given nodes which are also in the graph and returns an iterator over them.
        """
    def neighbors(self, n: int) -> object: 
        """
        Returns an iterable of incoming and outgoing nodes of n. Potentially with duplicates.
        """
    @typing.overload
    def number_of_edges(self) -> int: 
        """
        Returns the number of edges between two nodes.

        Returns edges count
        """
    @typing.overload
    def number_of_edges(self, arg0: int, arg1: int) -> int: ...
    def number_of_nodes(self) -> int: 
        """
        Returns the number of nodes in the graph.
        """
    def order(self) -> int: 
        """
        Returns the number of nodes in the graph.
        """
    def predecessors(self, n: int) -> object: 
        """
        Returns an iterable of follower nodes of n.
        """
    @typing.overload
    def remove_edge(self, u_for_edge: int, v_for_edge: int) -> None: ...
    @typing.overload
    def remove_edge(self, u_for_edge: int, v_for_edge: int, key: int) -> None: ...
    @typing.overload
    def remove_edges_from(self, ebunch: object) -> None: 
        """
        Removes all edges in supplied adjacency list (in a form of 2 or 3 columnar matrix) from the graph.

        Removes edges from members of the first array to members of the second array.
        """
    @typing.overload
    def remove_edges_from(self, us: object, vs: object, keys: object = None) -> None: ...
    def remove_node(self, v_to_remove: int) -> None: ...
    def remove_nodes_from(self, arg0: object) -> None: ...
    def reverse(self) -> None: ...
    def set_edge_attributes(self, values: object, name: typing.Optional[str] = None) -> None: ...
    def set_node_attributes(self, values: object, name: typing.Optional[str] = None) -> None: ...
    def size(self) -> int: 
        """
        Returns the number of attributed edges.
        """
    @typing.overload
    def subgraph(self) -> None: 
        """
        Returns a subgraph in a form of an adjacency list with 3 columns, where every edge (row) contains at least one vertex from the supplied list. Some edges may be duplicated.

        Returns a subgraph in a form of an adjacency list with 3 columns, where every edge (row) contains at least one vertex from the supplied list at a distance withing a given number `hops` from the supplied `n`.
        """
    @typing.overload
    def subgraph(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def subgraph(self, arg0: object) -> None: ...
    def successors(self, n: int) -> object: 
        """
        Returns an iterable of successor nodes of n.
        """
    def to_directed(self) -> None: ...
    def to_undirected(self) -> None: ...
    @property
    def allows_loops(self) -> bool:
        """
        :type: bool
        """
    @property
    def degree(self) -> DegreeView:
        """
        A DegreeView for the graph.

        :type: DegreeView
        """
    @property
    def edges(self) -> EdgesRange:
        """
        :type: EdgesRange
        """
    @property
    def in_degree(self) -> DegreeView:
        """
        A DegreeView with the number incoming edges for each Vertex.

        :type: DegreeView
        """
    @property
    def nodes(self) -> NodesRange:
        """
        A NodeView of the graph.

        :type: NodesRange
        """
    @property
    def out_degree(self) -> DegreeView:
        """
        A DegreeView with the number outgoing edges for each Vertex.

        :type: DegreeView
        """
    pass
class NodesRange():
    @typing.overload
    def __call__(self, data: bool = False) -> NodesRange: ...
    @typing.overload
    def __call__(self, data: str, default: object = None) -> NodesRange: ...
    def __iter__(self) -> nodes_stream_t: ...
    pass
class NodesStream():
    def __next__(self) -> object: ...
    pass
class Transaction():
    def __enter__(self) -> Transaction: ...
    def __exit__(self, arg0: object, arg1: object, arg2: object) -> bool: ...
    def __getitem__(self, collection: str) -> Collection: ...
    def __init__(self, db: DataBase, begin: bool = True, watch: bool = True, flush_writes: bool = False, snapshot: bool = False) -> None: ...
    def commit(self) -> None: ...
    @property
    def main(self) -> Collection:
        """
        :type: Collection
        """
    pass
def allows_loops(arg0: Network) -> bool:
    pass
def density(arg0: Network) -> float:
    pass
def from_dict(arg0: Collection, arg1: object) -> DataFrame:
    pass
def from_records(arg0: Collection, arg1: object) -> DataFrame:
    pass
def is_directed(arg0: Network) -> bool:
    pass
def is_multi(arg0: Network) -> bool:
    pass
def write_adjlist(G: Network, path: str, comments: str = '#', delimiter: str = ' ', encoding: str = 'utf-8') -> None:
    pass
