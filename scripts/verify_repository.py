"""Verify the publication files and preserved baseline computations without Torch."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class StripDocstrings(ast.NodeTransformer):
    def generic_visit(self,node):
        node=super().generic_visit(node)
        if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef)) and node.body:
            first=node.body[0]
            if isinstance(first,ast.Expr) and isinstance(first.value,ast.Constant) and isinstance(first.value.value,str): node.body=node.body[1:]
        return node

class SelectBaseline(ast.NodeTransformer):
    def visit_If(self,node):
        if ast.unparse(node.test)=="variant == 'baseline'": return [self.visit(x) for x in node.body]
        return self.generic_visit(node)

def parsed(path): return ast.parse(path.read_text(encoding='utf-8'))
def function(tree,name): return next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==name)
def eq(a,b): return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)

def main():
    manifest=json.loads((ROOT/'release_manifest.json').read_text())
    for name,value in manifest.items():
        got=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if got!=value: raise RuntimeError('Publication file changed: '+name)
    orig=ROOT/'provenance/original_runtime'
    original_model=StripDocstrings().visit(parsed(orig/'baseline_zxvad.py'))
    current_model=StripDocstrings().visit(parsed(ROOT/'src/baseline_zxvad.py'))
    assert eq(original_model,current_model),'Executable model/loss module changed'
    original_update=SelectBaseline().visit(copy.deepcopy(function(parsed(orig/'train.py'),'one_update')))
    current_update=function(parsed(ROOT/'src/train.py'),'one_update')
    assert eq(original_update,current_update),'Baseline update computations changed'
    a,b=parsed(orig/'common.py'),parsed(ROOT/'src/common.py')
    for name in ('frames','identity','read_frame','sample_batch'):
        assert eq(function(a,name),function(b,name)),'Input pipeline changed: '+name
    a,b=parsed(orig/'metrics.py'),parsed(ROOT/'src/metrics.py')
    for name in ('normalized','AUC'):
        assert eq(function(a,name),function(b,name)),'AUROC computation changed: '+name
    # After specializing constants, the complete evaluator remains the archived evaluator.
    original_eval=StripDocstrings().visit(ast.parse((orig/'evaluate.py').read_text().replace('ZXVAD_MNA_V1','ZXVAD_INDEPENDENT_LEGACY_V1')))
    current_eval=StripDocstrings().visit(parsed(ROOT/'src/evaluate.py'))
    assert eq(original_eval,current_eval),'Frame evaluation computations changed'
    print(f'Publication hashes ({len(manifest)} files), model, baseline update, input pipeline and evaluator: PASS')

if __name__=='__main__': main()
