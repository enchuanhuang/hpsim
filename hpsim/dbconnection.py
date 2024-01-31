"""
DBConnection class
separated from hpsim.py on 2023/03/14 by En-Chuan Huang
"""
import os
import shutil
import logging
import pathlib

import HPSim

def get_default_db_path()-> os.PathLike:
    p = os.path.abspath(__file__)
    p = os.path.dirname(p)
    p = os.path.dirname(p)
    p = os.path.join(p, "db")
    return p

DB_DIR = get_default_db_path()


class DBConnection():
    """An hpsim class for creating the database connection.
    The user must provide the following arguments to constructor:

    databases: an ordered python list containing the individual database 
               filenames to be used in the simulations. The database must be 
               ordered to represent the linac from upstream to downstream
    """
    dbconnection = ""
    DB_DIR = get_default_db_path()


    def __init__(self, db_dir, databases, 
                 libsql_dir: str= os.path.join(DB_DIR, "lib"), 
                 libsql_file: str="libsqliteext.so", tmp_dir="", copydb=True):
        """Init loads and attaches databases so those original functions are not
        separately available
        
        Arguments:
           db_dir (str): path of dir containing db files
           databases (list of str): ordered list of database filenames in correct sequence
           libsql_dir (str): path of directory that contains external sql lib
           libsql_file (str): name of libsqliteext.so file
           tmp_dir(str, optional): FIXME

        Returns:
           dbconnection object
        """
        self.dbconnection = ""
        self.db_paths = []
        db_num = 0
        db_name = "main"
        
        # copy all the databases so that we do not overwrite them
        abs_db_dir = os.path.abspath(db_dir)
        if tmp_dir == "":
            tmp_dir = os.path.join(abs_db_dir, "tmp")
        if os.path.isdir(tmp_dir) == False:
            os.mkdir(tmp_dir)
        for db in databases:
            db_path = os.path.join(abs_db_dir,db)
            tmp_db_path = os.path.join(tmp_dir, db)
            if copydb:
                shutil.copy2(db_path, tmp_db_path)
                logging.debug(f"Copy from {db_path} to {tmp_db_path}")
            else:
                print("\n\n\nWARNING:We are modifying the original dbs\n\n\n")

        # connection
        for db in databases:
            db_path = os.path.join(tmp_dir,db)
            self.db_paths.append(db_path)
            if db is databases[0]:
                self.dbconnection = HPSim.DBConnection(db_path)
            else:
                db_num += 1
                db_name = 'db' + str(db_num)
                self.dbconnection.attach_db(db_path, db_name)
        logging.debug("db_paths = " + ",".join(self.db_paths))
        # assign to class variable the HPSim DBConnection object for use elsewhere
        DBConnection.dbconnection = self.dbconnection
        libsql_path = os.path.join(os.path.abspath(libsql_dir), libsql_file)
        self.dbconnection.load_lib(libsql_path)
        self.clear_model_index()

    def print_dbs(self):
        """Prints names of datatbases"""
        self.dbconnection.print_dbs()

    def get_db_paths(self):
        """get full path of dbs"""
        return self.db_paths

    def print_libs(self):
        """Prints the database library"""
        self.dbconnection.print_libs()

    def clear_model_index(self):
        """Clears model index. Must be called once db connection established"""
        self.dbconnection.clear_model_index()

    def get_epics_channels(self):
        """Returns a list of all the EPICS PV's in the db's connected thru 
        dbconnection"""
        return self.dbconnection.get_epics_channels()