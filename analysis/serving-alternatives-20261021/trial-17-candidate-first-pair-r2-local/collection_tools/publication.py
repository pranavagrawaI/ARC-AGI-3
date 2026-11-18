"""Durable audit publication. A report alone is never an accepted output."""
import hashlib,json,os
from pathlib import Path

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def durable_write(path,value):
    path=Path(path)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def invalid(report,error):
    value=dict(report);value.update(status='INVALID_INCONCLUSIVE',U_resp=None,ratio=None,qualification='FEASIBILITY_ONLY',supplemental_complete=False,pass_=False)
    value['issues']=[*report.get('issues',[]),repr(error)]
    return value

def failed(destination,report,error,write=durable_write):
    destination=Path(destination);value=invalid(report,error)
    # Quarantine any completion marker if a final publication step failed.
    complete=destination/'AUDIT-COMPLETION.json'
    if complete.exists():
        try:complete.rename(destination/'AUDIT-COMPLETION.failed.json')
        except OSError:value['completion_quarantine_error']=True
    try:write(destination/'AUDIT-FAILURE.json',value)
    except Exception as persistence_error:value['failure_receipt_persistence_error']=repr(persistence_error)
    return value

def read_completed(destination):
    destination=Path(destination)
    try:
        if (destination/'AUDIT-FAILURE.json').exists():return invalid(json.loads((destination/'AUDIT-FAILURE.json').read_bytes()),ValueError('audit failure receipt exists'))
        completed=json.loads((destination/'AUDIT-COMPLETION.json').read_bytes())
        report=json.loads((destination/'SUPPLEMENTAL-AUDIT.json').read_bytes())
        if completed['schema']!='SUPPLEMENTAL_AUDIT_OUTPUT_COMPLETION_V1' or completed['report_sha256']!=digest(destination/'SUPPLEMENTAL-AUDIT.json') or completed['origins_sha256']!=digest(destination/'ORIGINS.json'):
            raise ValueError('audit output completion/hash mismatch')
        if report.get('output_completion_required') is not True or report.get('supplemental_complete') is not True:
            raise ValueError('audit output incomplete')
        return report
    except Exception as error:return invalid(dict(issues=[]),error)

def publish(destination,report,proof,origins,write=durable_write):
    destination=Path(destination)
    try:
        write(destination/'ORIGINS.json',origins)
        candidate=dict(report);candidate.update(proof,supplemental_complete=True,output_completion_required=True)
        write(destination/'SUPPLEMENTAL-AUDIT.json',candidate)
        completed=dict(schema='SUPPLEMENTAL_AUDIT_OUTPUT_COMPLETION_V1',
            report_sha256=digest(destination/'SUPPLEMENTAL-AUDIT.json'),origins_sha256=digest(destination/'ORIGINS.json'))
        pending=destination/'AUDIT-COMPLETION.pending.json';write(pending,completed)
        # Only acknowledge publication after the durable pending marker is complete.
        os.link(pending,destination/'AUDIT-COMPLETION.json');pending.unlink()
        fd=os.open(destination,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return read_completed(destination)
    except Exception as error:return failed(destination,report,error,write)
