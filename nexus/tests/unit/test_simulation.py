
import testing
from testing import value_eq,object_eq
from testing import TestFailed,failed
from testing import divert_nexus_log,restore_nexus_log


nexus_directories = dict()

def divert_nexus_directories():
    from nexus_base import nexus_core
    assert(len(nexus_directories)==0)
    nexus_directories['local']  = nexus_core.local_directory
    nexus_directories['remote'] = nexus_core.remote_directory
#end def divert_nexus_directories


def restore_nexus_directories():
    from nexus_base import nexus_core
    assert(set(nexus_directories.keys())==set(['local','remote']))
    nexus_core.local_directory  = nexus_directories['local'] 
    nexus_core.remote_directory = nexus_directories['remote']
    nexus_directories.clear()
#end def restore_nexus_directories


from generic import obj
from simulation import Simulation,SimulationInput,SimulationAnalyzer

class TestSimulationInput(SimulationInput):
    def __init__(self,*args,**kwargs):
        SimulationInput.__init__(self,*args,**kwargs)

        self.result_data = obj()
    #end def __init__

    def is_valid(self):
        return True
    #end def is_valid

    def read(self,filepath):
        None
    #end def read

    def write(self,filepath=None):
        None
    #end def write

    def read_text(self,text,filepath=None):
        None
    #end def read_text

    def write_text(self,filepath=None):
        None
    #end def write_text

    def incorporate_system(self,system):
        None
    #end def incorporate_system

    def return_system(self):
        self.not_implemented()
    #end def return_system
#end class TestSimulationInput


class TestSimulationAnalyzer(SimulationAnalyzer):
    def __init__(self,sim):
        self.analysis_performed = False
    #end def __init__

    def analyze(self):
        self.analysis_performed = True
    #end def analyze
#end class TestSimulationAnalyzer


class TestSimulation(Simulation):
    input_type    = TestSimulationInput
    analyzer_type = TestSimulationAnalyzer

    application_results = set(['quant1','quant2','quant3'])

    def check_sim_status(self):
        self.finished = True
    #end def check_sim_status

    def get_output_files(self):
        return []
    #end def get_output_files

    def check_result(self,result_name,sim):
        return result_name in self.application_results
    #end def check_result

    def get_result(self,result_name,sim):
        result = obj()
        result.name  = result_name
        result.simid = sim.simid
        return result
    #end def get_result

    def incorporate_result(self,result_name,result,sim):
        self.input.result_data[result.simid] = result.name
    #end def incorporate_result
#end class TestSimulation




get_sim_simulations = []

def get_sim(**kwargs):
    from machines import job
    from simulation import Simulation

    test_job = job(machine='ws1',app_command='test.x')

    n = len(get_sim_simulations)

    sim = Simulation(identifier='sim'+str(n),job=test_job,**kwargs)

    get_sim_simulations.append(sim)

    return sim
#end def get_sim


get_test_sim_simulations = []

def get_test_sim(**kwargs):
    from machines import job

    test_job = job(machine='ws1',app_command='test.x')

    n = len(get_test_sim_simulations)

    test_sim = TestSimulation(
        identifier = 'test_sim'+str(n),
        job        = test_job,
        **kwargs
        )

    get_test_sim_simulations.append(test_sim)

    return test_sim
#end def get_test_sim



def test_import():
    import simulation
    from simulation import Simulation,SimulationInput,SimulationAnalyzer
    from simulation import SimulationImage
    from simulation import NullSimulationInput,NullSimulationAnalyzer
    from simulation import GenericSimulation
    from simulation import SimulationInputTemplate
    from simulation import SimulationInputMultiTemplate
    from simulation import input_template,multi_input_template
    from simulation import generate_simulation
#end def test_import



def test_simulation_input():
    import os
    from generic import NexusError
    from simulation import SimulationInput

    tpath = testing.setup_unit_test_output_directory('simulation','test_simulation_input')

    # empty init
    si = SimulationInput()

    # write
    infile = os.path.join(tpath,'sim_input.in')
    wtext = 'simulation input'
    si.write_file_text(infile,wtext)
    assert(os.path.exists(infile))

    # read
    rtext = si.read_file_text(infile)
    assert(rtext==wtext)

    # virtuals
    virts = [
        si.is_valid,
        si.return_structure,
        (si.read_text,[None]),
        si.write_text,
        (si.incorporate_system,[None]),
        si.return_system,
        ]
    for v in virts:
        args = []
        if isinstance(v,tuple):
            v,args = v
        #end if
        try:
            v(*args)
            raise TestFailed
        except NexusError:
            None
        except TestFailed:
            failed(str(v))
        except Exception as e:
            failed(str(e))
        #end try
    #end for
#end def test_simulation_input



def test_simulation_analyzer():
    import os
    from generic import NexusError
    from simulation import SimulationAnalyzer

    # empty init
    try:
        SimulationAnalyzer()
        raise TestFailed
    except TestFailed:
        failed()
    except:
        None
    #end try

    # virtuals
    try:
        SimulationAnalyzer(None)
        raise TestFailed
    except NexusError:
        None
    except TestFailed:
        failed()
    except Exception as e:
        failed(str(e))
    #end try
#end def test_simulation_analyzer



def test_simulation_input_template():
    import os
    from string import Template
    from generic import obj,NexusError
    from simulation import SimulationInput
    from simulation import GenericSimulationInput
    from simulation import SimulationInputTemplate
    from simulation import input_template

    tpath = testing.setup_unit_test_output_directory('simulation','test_simulation_input_template')


    # empty init
    si_empty = input_template()
    assert(isinstance(si_empty,SimulationInput))
    assert(isinstance(si_empty,GenericSimulationInput))
    assert(isinstance(si_empty,SimulationInputTemplate))

    si_empty_ref = obj(
        template      = None,
        keywords      = set(),
        values        = obj(),
        allow_not_set = set(),
        )

    assert(len(si_empty)==4)
    assert(object_eq(si_empty.to_obj(),si_empty_ref))


    # template reference data
    template_text = '''
a     = "$a"
b     = $b
file1 = "$file.$ext1"
file2 = "$file.$ext2"
'''

    template_filepath = os.path.join(tpath,'template_file.txt')

    open(template_filepath,'w').write(template_text)


    # read
    si_read = input_template(template_filepath)

    assert(isinstance(si_read.template,Template))
    assert(si_read.keywords==set(['a','b','ext1','ext2','file']))


    # assign
    si_assign = input_template()
    try:
        si_assign.assign(b=1)
        raise TestFailed
    except NexusError:
        None
    except TestFailed:
        failed()
    except Exception as e:
        failed(str(e))
    #end try

    si_assign = input_template(template_filepath)
    try:
        si_assign.assign(c=1)
        raise TestFailed
    except NexusError:
        None
    except TestFailed:
        failed()
    except Exception as e:
        failed(str(e))
    #end try

    values = obj(
        a    = 'name',
        b    = 1,
        file = 'my_file',
        ext1 = 'txt',
        ext2 = 'dat',
        )

    si_assign.assign(**values)

    assert(object_eq(si_assign.values,values))


    # write
    def try_write(si):
        try:
            si.write()
            raise TestFailed
        except NexusError:
            None
        except TestFailed:
            failed()
        except Exception as e:
            failed(str(e))
        #end try
    #end def try_write

    si_write = input_template()
    try_write(si_write)

    si_write = input_template(template_filepath)
    try_write(si_write)

    si_write.assign(b=1)
    try_write(si_write)


    text_ref = '''
a     = "name"
b     = 1
file1 = "my_file.txt"
file2 = "my_file.dat"
'''

    si_write.assign(**values)
    text = si_write.write()
    assert(text==text_ref)
    
    input_filepath = os.path.join(tpath,'input_file.txt')
    si_write.write(input_filepath)
    assert(open(input_filepath,'r').read()==text_ref)

#end def test_simulation_input_template



def test_simulation_input_multi_template():
    import os
    from string import Template
    from generic import obj,NexusError
    from simulation import SimulationInput
    from simulation import GenericSimulationInput
    from simulation import SimulationInputMultiTemplate
    from simulation import multi_input_template

    tpath = testing.setup_unit_test_output_directory('simulation','test_simulation_input_multi_template')

    # make template files
    template1_filepath = os.path.join(tpath,'template1.txt')
    template2_filepath = os.path.join(tpath,'template2.txt')
    template3_filepath = os.path.join(tpath,'template3.txt')

    open(template1_filepath,'w').write('''
name = "$name"
a    = $a
''')
    open(template2_filepath,'w').write('''
name = "$name"
b    = $b
''')
    open(template3_filepath,'w').write('''
name = "$name"
c    = $c
''')

    input1_filepath = os.path.join(tpath,'input_file1.txt')
    input2_filepath = os.path.join(tpath,'input_file2.txt')
    input3_filepath = os.path.join(tpath,'input_file3.txt')


    # empty init
    si_empty = multi_input_template()

    assert(isinstance(si_empty,SimulationInput))
    assert(isinstance(si_empty,GenericSimulationInput))
    assert(isinstance(si_empty,SimulationInputMultiTemplate))

    si_empty_ref = obj(
        filenames = obj(),
        )

    assert(len(si_empty)==1)
    assert(object_eq(si_empty.to_obj(),si_empty_ref))


    # filename init
    filenames = obj(
        input1 = 'input_file1.txt',
        input2 = 'input_file2.txt',
        input3 = 'input_file3.txt',
        )

    si = multi_input_template(**filenames)

    assert(len(si)==1)
    assert(len(si.filenames)==3)
    assert(object_eq(si.filenames,filenames))

    
    # init read
    si_init = multi_input_template(
        input1 = ('input_file1.txt',template1_filepath),
        input2 = ('input_file2.txt',template2_filepath),
        input3 = ('input_file3.txt',template3_filepath),
        )
    si = si_init
    assert(len(si)==4)
    assert(len(si.filenames)==3)
    assert(object_eq(si.filenames,filenames))
    keywords_ref = dict(
        input1 = set(['a', 'name']),
        input2 = set(['b', 'name']),
        input3 = set(['c', 'name']),
        )
    for name,keyword_set in keywords_ref.items():
        assert(name in si)
        sit = si[name]
        assert(sit.keywords==keyword_set)
        assert(isinstance(sit.template,Template))
        assert(object_eq(sit.values,obj()))
        assert(sit.allow_not_set==set())
    #end for


    # write
    write_ref = obj(
        input1 = '''
name = "name1"
a    = 1
''',
        input2 = '''
name = "name2"
b    = 2
''',
        input3 = '''
name = "name3"
c    = 3
''',
        )

    si_write = multi_input_template(
        input1 = ('input_file1.txt',template1_filepath),
        input2 = ('input_file2.txt',template2_filepath),
        input3 = ('input_file3.txt',template3_filepath),
        )
    si_write.input1.assign(
        name = 'name1',
        a    = 1,
        )
    si_write.input2.assign(
        name = 'name2',
        b    = 2,
        )
    si_write.input3.assign(
        name = 'name3',
        c    = 3,
        )
    assert(object_eq(si_write.write(),write_ref))

    si_write.write(input1_filepath)
    assert(os.path.exists(input1_filepath))
    assert(os.path.exists(input2_filepath))
    assert(os.path.exists(input3_filepath))


    # read
    si_read = multi_input_template(**filenames)

    si_read.read(input1_filepath)

    si = si_read
    assert(len(si)==4)
    assert(len(si.filenames)==3)
    assert(object_eq(si.filenames,filenames))
    for name,keyword_set in keywords_ref.items():
        assert(name in si)
        sit = si[name]
        assert(sit.keywords==set())
        assert(isinstance(sit.template,Template))
        assert(object_eq(sit.values,obj()))
        assert(sit.allow_not_set==set())
    #end for
    assert(object_eq(si_read.write(),write_ref))

#end def test_simulation_input_multi_template



def test_code_name():
    from simulation import Simulation

    cn = Simulation.code_name()
    assert(isinstance(cn,str))
    assert(' ' not in cn)
#end def test_code_name



def test_init():
    from generic import obj
    from machines import job,Job
    from simulation import Simulation,SimulationInput

    # empty init, tests set(), set_directories(), set_files()
    se = Simulation()

    se_ref = obj(
        analyzed             = False,
        analyzer_image       = 'analyzer.p',
        app_name             = 'simapp',
        app_props            = ['serial'],
        block                = False,
        block_subcascade     = False,
        bundleable           = True,
        bundled              = False,
        bundler              = None,
        created_directories  = False,
        dependency_ids       = set([]),
        errfile              = 'sim.err',
        failed               = False,
        fake_sim             = False,
        files                = set([]),
        finished             = False,
        force_restart        = False,
        force_write          = False,
        got_dependencies     = False,
        got_output           = False,
        identifier           = 'sim',
        image_dir            = 'sim_sim',
        imlocdir             = './runs/sim_sim',
        imremdir             = './runs/sim_sim',
        imresdir             = './results/runs/sim_sim',
        infile               = 'sim.in',
        input_image          = 'input.p',
        locdir               = './runs/',
        job                  = None,
        loaded               = False,
        ordered_dependencies = [],
        outfile              = 'sim.out',
        outputs              = None,
        path                 = '',
        process_id           = None,
        restartable          = False,
        remdir               = './runs/',
        resdir               = './results/runs/',
        sent_files           = False,
        setup                = False,
        sim_image            = 'sim.p',
        simlabel             = None,
        skip_submit          = False,
        subcascade_finished  = False,
        submitted            = False,
        system               = None,
        wait_ids             = set([]),
        dependencies         = obj(),
        dependents           = obj(),
        input                = SimulationInput(),
        )

    assert(object_eq(se.obj(se_ref.keys()),se_ref))
    assert(isinstance(se.simid,int))
    assert(se.simid>=0)
    assert(se.simid<Simulation.sim_count)

    Simulation.clear_all_sims()
    assert(len(Simulation.all_sims)==0)
    assert(len(Simulation.sim_directories)==0)
    assert(Simulation.sim_count==0)


    # make a test job
    test_job = job(machine='ws1',app_command='test.x')

    
    # minimal non-empty init, tests init_job()
    sm = Simulation(job=test_job)

    sm_ref = se_ref.copy()
    del sm_ref.job
    assert(object_eq(sm.obj(sm_ref.keys()),sm_ref))
    assert(isinstance(se.simid,int))
    assert(se.simid>=0)
    assert(se.simid<Simulation.sim_count)
    assert(isinstance(sm.job,Job))
    assert(id(sm.job)!=id(test_job))


    # initialization tests for set_directories()
    # two sims in same directory w/ same identifier should fail
    try:
        s1 = Simulation(
            identifier = 'same_identifier',
            path       = 'same_directory',
            job        = test_job,
            )
        s2 = Simulation(
            identifier = 'same_identifier',
            path       = 'same_directory',
            job        = test_job,
            )
        raise TestFailed
    except TestFailed:
        failed()
    except:
        None
    #end try

    # two sims in same directory w/ different identifiers should be ok
    s1 = Simulation(
        identifier = 'identifier1',
        path       = 'same_directory',
        job        = test_job,
        )
    s2 = Simulation(
        identifier = 'identifier2',
        path       = 'same_directory',
        job        = test_job,
        )

    # two sims in different directories w/ same identifier should be ok
    s1 = Simulation(
        identifier = 'same_identifier',
        path       = 'directory1',
        job        = test_job,
        )
    s2 = Simulation(
        identifier = 'same_identifier',
        path       = 'directory2',
        job        = test_job,
        )

    Simulation.clear_all_sims()

#end def test_init



def test_virtuals():
    from generic import NexusError
    from simulation import Simulation

    s = Simulation()

    virts = [
        (s.check_result,[None,None]),
        (s.get_result,[None,None]),
        (s.incorporate_result,[None,None,None]),
        s.app_command,
        s.check_sim_status,
        s.get_output_files,
        ]
    for v in virts:
        args = []
        if isinstance(v,tuple):
            v,args = v
        #end if
        try:
            v(*args)
            raise TestFailed
        except NexusError:
            None
        except TestFailed:
            failed(str(v))
        except Exception as e:
            failed(str(e))
        #end try
    #end for

    vacuous_virts = [
        s.propagate_identifier,
        s.pre_init,
        s.post_init,
        s.pre_create_directories,
        s.write_prep,
        (s.pre_write_inputs,[None]),
        (s.pre_send_files,[None]),
        s.post_submit,
        s.pre_check_status,
        (s.post_analyze,[None]),
        ]
    for v in vacuous_virts:
        args = []
        if isinstance(v,tuple):
            v,args = v
        #end if
        v(*args)
    #end for

    Simulation.clear_all_sims()

#end def test_virtuals



def test_reset_indicators():
    from simulation import Simulation

    indicators = '''
        got_dependencies
        setup     
        sent_files
        submitted 
        finished  
        failed    
        got_output
        analyzed  
        '''.split()

    s = Simulation()

    for i in indicators:
        s[i] = True
    #end for

    s.reset_indicators()

    for i in indicators:
        ind = s[i]
        assert(isinstance(ind,bool))
        assert(not ind)
    #end for

    Simulation.clear_all_sims()
#end def test_reset_indicators



def test_indicator_checks():
    from machines import job
    from simulation import Simulation

    def complete(sim):
        sim.setup      = True
        sim.sent_files = True
        sim.submitted  = True
        sim.finished   = True
        sim.got_output = True
        sim.analyzed   = True
        sim.failed     = False
    #end def complete

    # test completed()
    s = Simulation()
    assert(not s.completed())
    complete(s)
    assert(s.completed())
    s.reset_indicators()
    Simulation.clear_all_sims()

    # test ready() and active()
    test_job = job(machine='ws1',app_command='test.x')

    simdeps = []
    for i in range(5):
        s = Simulation(identifier='id'+str(i),job=test_job)
        complete(s)
        simdeps.append((s,'other'))
    #end for

    s = Simulation(job=test_job,dependencies=simdeps)
    assert(s.ready())
    assert(s.active())
    s.submitted = True
    assert(not s.ready())
    assert(s.active())

    Simulation.clear_all_sims()

#end def test_indicator_checks



def test_create_directories():
    import os
    from nexus_base import nexus_core
    from simulation import Simulation

    tpath = testing.setup_unit_test_output_directory('simulation','test_create_directories')

    divert_nexus_directories()

    nexus_core.local_directory  = tpath
    nexus_core.remote_directory = tpath

    s = Simulation()

    assert(not os.path.exists(s.locdir))
    assert(not os.path.exists(s.imlocdir))
    assert(not s.created_directories)

    s.create_directories()

    assert(os.path.exists(s.locdir))
    assert(os.path.exists(s.imlocdir))
    assert(s.created_directories)

    restore_nexus_directories()

    Simulation.clear_all_sims()
#end def test_create_directories



def test_file_text():
    import os
    from nexus_base import nexus_core
    from simulation import Simulation

    tpath = testing.setup_unit_test_output_directory('simulation','test_create_directories')

    divert_nexus_directories()

    nexus_core.local_directory  = tpath
    nexus_core.remote_directory = tpath

    s = Simulation()
    s.create_directories()

    outfile = os.path.join(s.locdir,s.outfile)
    errfile = os.path.join(s.locdir,s.errfile)

    out_text = 'output'
    err_text = 'error'

    open(outfile,'w').write(out_text)
    open(errfile,'w').write(err_text)

    assert(s.outfile_text()==out_text)
    assert(s.errfile_text()==err_text)

    restore_nexus_directories()

    Simulation.clear_all_sims()
#end def test_file_text



def check_dependency_objects(*sims,**kwargs):
    from generic import obj
    from simulation import Simulation
    empty    = kwargs.get('empty',False)
    wait_ids = kwargs.get('wait_ids',True)
    if len(sims)==1 and isinstance(sims[0],list):
        sims = sims[0]
    #end if
    for sim in sims:
        if empty:
            assert(value_eq(sim.ordered_dependencies,[]))
            assert(isinstance(sim.dependencies,obj))
            assert(len(sim.dependencies)==0)
            assert(sim.dependency_ids==set())
            assert(sim.wait_ids==set())
        else:
            # check dependencies object
            for simid,dep in sim.dependencies.iteritems():
                assert(isinstance(simid,int))
                assert(isinstance(dep,obj))
                assert('result_names' in dep)
                assert('results' in dep)
                assert('sim' in dep)
                assert(len(dep)==3)
                assert(isinstance(dep.sim,Simulation))
                assert(simid==dep.sim.simid)
                assert(isinstance(dep.result_names,list))
                for name in dep.result_names:
                    assert(isinstance(name,str))
                #end for
                assert(isinstance(dep.results,obj))
                assert(len(dep.results)==0)
            #end for
            # check ordered_dependencies object
            for dep in sim.ordered_dependencies:
                dep2 = sim.dependencies[dep.sim.simid]
                assert(id(dep2)==id(dep))
            #end for
            # check dependents object
            for dsimid,dsim in sim.dependents.items():
                assert(isinstance(dsimid,int))
                assert(isinstance(dsim,Simulation))
                assert(dsimid==dsim.simid)
                assert(sim.simid in dsim.dependency_ids)
                assert(sim.simid in dsim.dependencies)
                assert(id(sim)==id(dsim.dependencies[sim.simid].sim))
                found = False
                for s in dsim.ordered_dependencies:
                    found |= id(s.sim)==id(sim)
                #end for
                assert(found)
            #end for
            # check dependency_ids
            for simid in sim.dependency_ids:
                assert(isinstance(simid,int))
                assert(simid in sim.dependencies)
            #end for
            # check wait_ids
            if wait_ids:
                assert(sim.wait_ids==sim.dependency_ids)
            #end if
        #end if
    #end if
#end def check_dependency_objects



def check_dependency(sim2,sim1,quants=['other'],only=False,objects=False):
    # sim2 depends on sim1 for all quantities
    if objects:
        check_dependency_objects(sim1)
        check_dependency_objects(sim2)
    #end if
    assert(sim2.simid in sim1.dependents)
    assert(id(sim1.dependents[sim2.simid])==id(sim2))
    assert(sim1.simid in sim2.dependency_ids)
    assert(sim1.simid in sim2.dependencies)
    assert(id(sim2.dependencies[sim1.simid].sim)==id(sim1))
    assert(set(sim2.dependencies[sim1.simid].result_names)==set(quants))
    if only:
        assert(len(sim1.dependents)==1)
        assert(len(sim2.dependency_ids)==1)
        assert(len(sim2.dependencies)==1)
        assert(len(sim2.ordered_dependencies)==1)
    #end if
#end def check_dependency


def test_depends():
    from generic import NexusError
    from simulation import Simulation

    # single dependency, single quantity
    s1 = get_sim()
    s2 = get_sim()

    check_dependency_objects(s1,empty=True)
    check_dependency_objects(s2,empty=True)

    s2.depends(s1,'other')

    check_dependency(s2,s1,objects=True,only=True)
    del s1,s2

    s1 = get_sim()
    s2 = get_sim()
    s2.depends((s1,'other'))
    check_dependency(s2,s1,objects=True,only=True)
    del s1,s2

    s1 = get_sim()
    s2 = get_sim(
        dependencies = (s1,'other'),
        )
    check_dependency(s2,s1,objects=True,only=True)
    del s1,s2

    s1 = get_sim()
    s2 = get_sim(
        dependencies = [(s1,'other')],
        )
    check_dependency(s2,s1,objects=True,only=True)
    del s1,s2


    # single dependency, multiple quantities
    s1 = get_test_sim()
    s2 = get_test_sim()

    quants = ['quant1','quant2','quant3']

    check_dependency_objects(s1,empty=True)
    check_dependency_objects(s2,empty=True)

    s2.depends(s1,'quant1','quant2','quant3')

    check_dependency(s2,s1,quants,objects=True,only=True)
    del s1,s2

    s1 = get_test_sim()
    s2 = get_test_sim()
    s2.depends(s1,'quant1')
    s2.depends(s1,'quant2')
    s2.depends(s1,'quant3')
    check_dependency(s2,s1,quants,objects=True,only=True)
    del s1,s2

    s1 = get_test_sim()
    s2 = get_test_sim()
    s2.depends(s1,'quant1','quant2')
    s2.depends(s1,'quant3')
    check_dependency(s2,s1,quants,objects=True,only=True)
    del s1,s2

    s1 = get_test_sim()
    s2 = get_test_sim()
    s2.depends((s1,'quant1','quant2','quant3'))
    check_dependency(s2,s1,quants,objects=True,only=True)
    del s1,s2

    s1 = get_test_sim()
    s2 = get_test_sim()
    s2.depends((s1,'quant1'))
    s2.depends((s1,'quant2'))
    s2.depends((s1,'quant3'))
    check_dependency(s2,s1,quants,objects=True,only=True)
    del s1,s2

    s1 = get_test_sim()
    s2 = get_test_sim(
        dependencies = (s1,'quant1','quant2','quant3'),
        )
    check_dependency(s2,s1,quants,objects=True,only=True)
    del s1,s2

    s1 = get_test_sim()
    s2 = get_test_sim(
        dependencies = [(s1,'quant1','quant2','quant3')],
        )
    check_dependency(s2,s1,quants,objects=True,only=True)
    del s1,s2

    s1 = get_test_sim()
    s2 = get_test_sim(
        dependencies = [
            (s1,'quant1'),
            (s1,'quant2'),
            (s1,'quant3'),
            ],
        )
    check_dependency(s2,s1,quants,objects=True,only=True)
    del s1,s2


    # multiple dependencies
    s11 = get_test_sim()
    s12 = get_test_sim()
    s13 = get_test_sim()

    s21 = get_test_sim(
        dependencies = [
            (s11,'quant1'),
            (s12,'quant2'),
            ]
        )
    s22 = get_test_sim(
        dependencies = [
            (s12,'quant2'),
            (s13,'quant3'),
            ]
        )

    s31 = get_test_sim(
        dependencies = [
            (s21,'quant1'),
            (s22,'quant2'),
            ]
        )
    s32 = get_test_sim(
        dependencies = [
            (s21,'quant1'),
            (s22,'quant2'),
            ]
        )
    s33 = get_test_sim(
        dependencies = [
            (s21,'quant1'),
            (s22,'quant2'),
            ]
        )

    s41 = get_test_sim(
        dependencies = [
            (s11,'quant1'),
            (s22,'quant2'),
            (s32,'quant3'),
            ]
        )

    check_dependency_objects(s11,s12,s13,s21,s22,s31,s32,s33,s41)

    check_dependency(s21,s11,['quant1'])
    check_dependency(s21,s12,['quant2'])

    check_dependency(s22,s12,['quant2'])
    check_dependency(s22,s13,['quant3'])

    check_dependency(s31,s21,['quant1'])
    check_dependency(s31,s22,['quant2'])

    check_dependency(s32,s21,['quant1'])
    check_dependency(s32,s22,['quant2'])

    check_dependency(s33,s21,['quant1'])
    check_dependency(s33,s22,['quant2'])

    check_dependency(s41,s11,['quant1'])
    check_dependency(s41,s22,['quant2'])
    check_dependency(s41,s32,['quant3'])

    del s11,s12,s13,s21,s22,s31,s32,s33,s41


    # fail when dependency does not exist
    try:
        s1 = get_sim()
        s2 = get_sim(
            dependencies = [(s1,'quant1')],
            )
        raise TestFailed
    except NexusError:
        None
    except:
        failed()
    #end try

    try:
        s1 = get_sim()
        s2 = get_sim(
            dependencies = [(s1,'other','quant2')],
            )
        raise TestFailed
    except NexusError:
        None
    except:
        failed()
    #end try

    try:
        s1 = get_test_sim()
        s2 = get_test_sim(
            dependencies = [(s1,'quant1','apple')],
            )
        raise TestFailed
    except NexusError:
        None
    except:
        failed()
    #end try

    Simulation.clear_all_sims()
#end def test_depends



def test_undo_depends():
    from simulation import Simulation

    # single dependency, single quantity
    s1 = get_sim()
    s2 = get_sim()

    check_dependency_objects(s1,empty=True)
    check_dependency_objects(s2,empty=True)

    s2.depends(s1,'other')

    check_dependency(s2,s1,objects=True,only=True)

    s2.undo_depends(s1)

    check_dependency_objects(s1,empty=True)
    check_dependency_objects(s2,empty=True)

    Simulation.clear_all_sims()
#end def test_undo_depends



def test_has_generic_input():
    from simulation import Simulation
    from simulation import SimulationInput,GenericSimulationInput

    s = get_sim()
    assert(not s.has_generic_input())
    del s

    class GenInput(SimulationInput,GenericSimulationInput):
        None
    #end class GenInput

    s = get_sim(
        input = GenInput(),
        )
    assert(s.has_generic_input())
    del s

    Simulation.clear_all_sims()
#end def test_has_generic_input



def test_check_dependencies():
    from generic import obj,NexusError
    from simulation import Simulation
    from simulation import SimulationInput,GenericSimulationInput

    result = obj()
    result.dependencies_satisfied = True

    s11 = get_test_sim()
    s12 = get_test_sim()
    s13 = get_test_sim()

    s21 = get_test_sim(
        dependencies = [
            (s11,'quant1'),
            (s12,'quant2'),
            ]
        )
    s22 = get_test_sim(
        dependencies = [
            (s12,'quant2'),
            (s13,'quant3'),
            ]
        )

    s31 = get_test_sim(
        dependencies = [
            (s21,'quant1'),
            (s22,'quant2'),
            ]
        )
    s32 = get_test_sim(
        dependencies = [
            (s21,'quant1'),
            (s22,'quant2'),
            ]
        )
    s33 = get_test_sim(
        dependencies = [
            (s21,'quant1'),
            (s22,'quant2'),
            ]
        )

    s41 = get_test_sim(
        dependencies = [
            (s11,'quant1'),
            (s22,'quant2'),
            (s32,'quant3'),
            ]
        )

    sims = [s11,s12,s13,s21,s22,s31,s32,s33,s41]
    for s in sims:
        s.check_dependencies(result)
    #end for
    assert(result.dependencies_satisfied)


    # non-existent dependency
    try:
        s  = get_test_sim()
        s2 = get_test_sim(dependencies=((s,'nonexistent')))
        raise TestFailed
    except NexusError:
        None
    except TestFailed:
        failed()
    except Exception as e:
        failed(str(e))
    #end try


    # existent dependency but generic input
    divert_nexus_log()
    class GenInput(SimulationInput,GenericSimulationInput):
        None
    #end class GenInput

    s = get_test_sim(
        input = GenInput(),
        )

    s2 = get_test_sim(
        input = GenInput(),
        dependencies = (s,'quant1')
        )

    result = obj(dependencies_satisfied=True)
    s.check_dependencies(result)
    assert(result.dependencies_satisfied)

    try:
        s2.check_dependencies(result)
    except NexusError:
        None
    except TestFailed:
        failed()
    except Exception as e:
        failed(str(e))
    #end try
    restore_nexus_log()

    Simulation.clear_all_sims()
#end def test_check_dependencies



def test_get_dependencies():
    from generic import obj
    from simulation import Simulation

    simdeps = obj()

    deps = []
    s11 = get_test_sim()
    simdeps[s11.simid] = deps

    s12 = get_test_sim()
    simdeps[s12.simid] = deps

    s13 = get_test_sim()
    simdeps[s13.simid] = deps

    deps = [
        (s11,'quant1'),
        (s12,'quant2'),
        ]
    s21 = get_test_sim(dependencies=deps)
    simdeps[s21.simid] = deps

    deps = [
        (s12,'quant2'),
        (s13,'quant3'),
        ]
    s22 = get_test_sim(dependencies=deps)
    simdeps[s22.simid] = deps

    dependencies = [
        (s21,'quant1'),
        (s22,'quant2'),
        ]
    s31 = get_test_sim(dependencies=deps)
    simdeps[s31.simid] = deps

    deps = [
        (s21,'quant1'),
        (s22,'quant2'),
        ]
    s32 = get_test_sim(dependencies=deps)
    simdeps[s32.simid] = deps

    deps = [
        (s21,'quant1'),
        (s22,'quant2'),
        ]
    s33 = get_test_sim(dependencies=deps)
    simdeps[s33.simid] = deps

    deps = [
        (s11,'quant1'),
        (s22,'quant2'),
        (s32,'quant3'),
        ]
    s41 = get_test_sim(dependencies=deps)
    simdeps[s41.simid] = deps

    sims = [s11,s12,s13,s21,s22,s31,s32,s33,s41]
    assert(len(simdeps)==len(sims))
    for s in sims:
        assert(not s.got_dependencies)
        assert(len(s.input.result_data)==0)
        s.get_dependencies()
        resdata = s.input.result_data
        deps = simdeps[s.simid]
        for sim,resname in deps:
            assert(sim.simid in resdata)
            assert(resdata[sim.simid]==resname)
        #end for
    #end for

    Simulation.clear_all_sims()
#end def test_get_dependencies



def test_downstream_simids():
    from generic import obj
    from simulation import Simulation

    s11 = get_test_sim()
    s12 = get_test_sim()
    s13 = get_test_sim()

    s21 = get_test_sim(
        dependencies = [
            (s11,'quant1'),
            (s12,'quant2'),
            ]
        )
    s22 = get_test_sim(
        dependencies = [
            (s12,'quant2'),
            (s13,'quant3'),
            ]
        )

    s31 = get_test_sim(
        dependencies = [
            (s21,'quant1'),
            (s22,'quant2'),
            ]
        )
    s32 = get_test_sim(
        dependencies = [
            (s21,'quant1'),
            (s22,'quant2'),
            ]
        )
    s33 = get_test_sim(
        dependencies = [
            (s21,'quant1'),
            (s22,'quant2'),
            ]
        )

    s41 = get_test_sim(
        dependencies = [
            (s11,'quant1'),
            (s22,'quant2'),
            (s32,'quant3'),
            ]
        )

    sims = obj(
        s11 = s11,
        s12 = s12,
        s13 = s13,
        s21 = s21,
        s22 = s22,
        s31 = s31,
        s32 = s32,
        s33 = s33,
        s41 = s41,
        )

    downstream_sims = obj(
        s11 = [s21,s31,s32,s33,s41],
        s12 = [s21,s22,s31,s32,s33,s41],
        s13 = [s22,s31,s32,s33,s41],
        s21 = [s31,s32,s33,s41],
        s22 = [s31,s32,s33,s41],
        s31 = [],
        s32 = [s41],
        s33 = [],
        s41 = [],
        )

    n = 0
    for sname in sorted(sims.keys()):
        s = sims[sname]
        ds_ids = s.downstream_simids()
        ds_ids_ref = set([sd.simid for sd in downstream_sims[sname]])
        assert(ds_ids==ds_ids_ref)
        n+=1
    #end for
    assert(n==9)

    Simulation.clear_all_sims()
#end def test_downstream_simids



def test_copy_file():
    import os
    from simulation import Simulation

    tpath = testing.setup_unit_test_output_directory('simulation','test_copy_file')
    
    opath = os.path.join(tpath,'other')
    if not os.path.exists(opath):
        os.makedirs(opath)
    #end if

    file1 = os.path.join(tpath,'file.txt')
    file2 = os.path.join(opath,'file.txt')

    open(file1,'w').write('text')
    assert(os.path.exists(file1))

    s = get_sim()

    s.copy_file(file1,opath)

    assert(os.path.exists(file2))
    assert(open(file2,'r').read().strip()=='text')
    
    Simulation.clear_all_sims()
#end def test_copy_file



def test_save_load_image():
    import os
    from generic import obj
    from nexus_base import nexus_core
    from simulation import Simulation,SimulationImage

    tpath = testing.setup_unit_test_output_directory('simulation','test_save_load_image')

    divert_nexus_directories()

    nexus_core.local_directory  = tpath
    nexus_core.remote_directory = tpath

    nsave = 30
    nload = 22

    assert(len(SimulationImage.save_fields)==nsave)
    assert(len(SimulationImage.load_fields)==nload)
    assert(len(SimulationImage.save_only_fields&SimulationImage.load_fields)==0)

    sim = get_sim()

    sim.create_directories()

    sim.save_image()

    imagefile = os.path.join(sim.imlocdir,sim.sim_image)
    assert(os.path.exists(imagefile))

    image = obj()
    image.load(imagefile)
    assert(len(image)==nsave)
    for field in SimulationImage.save_fields:
        assert(field in image)
        assert(field in sim)
        assert(value_eq(image[field],sim[field]))
    #end for

    orig = obj()
    for field in SimulationImage.load_fields:
        orig[field] = sim[field]
        del sim[field]
    #end for
    sim.sim_image = orig.sim_image
    sim.load_image()
    for field in SimulationImage.load_fields:
        assert(field in sim)
        assert(value_eq(sim[field],orig[field]))
    #end for

    restore_nexus_directories()

    Simulation.clear_all_sims()
#end def test_save_load_image



def test_load_analyzer_image():
    import os
    from nexus_base import nexus_core
    from simulation import Simulation

    tpath = testing.setup_unit_test_output_directory('simulation','test_save_load_analyzer_image')

    divert_nexus_directories()

    nexus_core.local_directory  = tpath
    nexus_core.remote_directory = tpath

    sim = get_test_sim()

    if not os.path.exists(sim.imresdir):
        os.makedirs(sim.imresdir)
    #end if

    analyzer_file = os.path.join(sim.imresdir,sim.analyzer_image)

    a = sim.analyzer_type(None)
    assert(not a.analysis_performed)
    a.analyze()
    assert(a.analysis_performed)
    a.save(analyzer_file)
    assert(os.path.exists(analyzer_file))

    a2 = sim.load_analyzer_image()
    assert(isinstance(a2,sim.analyzer_type))
    assert(a2.analysis_performed)
    assert(object_eq(a2,a))

    restore_nexus_directories()

    Simulation.clear_all_sims()
#end def test_load_analyzer_image



def test_save_attempt():
    import os
    from nexus_base import nexus_core
    from simulation import Simulation

    tpath = testing.setup_unit_test_output_directory('simulation','test_save_attempt')

    divert_nexus_directories()

    nexus_core.local_directory  = tpath
    nexus_core.remote_directory = tpath

    sim = get_test_sim()

    sim.create_directories()

    files = (sim.infile,sim.outfile,sim.errfile)

    assert(sim.attempt_files()==files)
    for file in files:
        open(os.path.join(sim.locdir,file),'w').write('made an attempt')
    #end for

    attempt_dir = os.path.join(sim.locdir,'{}_attempt1'.format(sim.identifier))
    assert(not os.path.exists(attempt_dir))
    sim.save_attempt()
    print attempt_dir
    assert(os.path.exists(attempt_dir))
    for file in files:
        assert(not os.path.exists(os.path.join(sim.locdir,file)))
        assert(os.path.exists(os.path.join(attempt_dir,file)))
    #end for

    restore_nexus_directories()

    Simulation.clear_all_sims()
#end def test_save_attempt