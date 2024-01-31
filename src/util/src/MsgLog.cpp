/*------------------------------------------------------
  MsgLog

  NOTE: NEED TO MAKE THIS WORK ON MACOSX
  (The long comment output does not appear on the mac)

  Author: J.L. Klay (CalPoly)
  Date: 29-Apr-2008

-------------------------------------------------------*/

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <cstdarg>
#include <cstring>
#include <memory>
#include <unistd.h>

#include "MsgLog.h"


MsgLog* MsgLog::fgInstance = 0x0;
bool MsgLog::fgDebugEnabled = true;


std::string get_filename(const char *path)
{
  std::string fullpath(path);
  std::string basename = fullpath.substr(fullpath.find_last_of("/") + 1);
  return basename;
}


//_____________________________________________________________________________
MsgLog::MsgLog() :
  fGlobalLogLevel(kInfo),
  fPrintRepetitions(true),
  fRepetitions(0),
  fLastType(0),
  fLastMessage(),
  fLastFunction(),
  fLastFile(),
  fLastLine(0)
{
  //default constructor

  for (int iType = kFatal; iType < kMaxType; iType++) {
    fOutputTypes[iType]   = 0;
    fFileNames[iType]     = "";
    fOutputFiles[iType]   = NULL;
    fOutputStreams[iType] = NULL;
    fCallBacks[iType]     =NULL;

    fPrintType[iType]     = true;
    fPrintLocation[iType] = (iType == kDebug);
    fPrintPrefix[iType]   = (iType == kDebug);
  }

  // replace the previous instance by this one
  if (fgInstance) delete fgInstance;
  fgInstance = this;

  //read the environment settings
  ReadEnvSettings();

}

//_____________________________________________________________________________
MsgLog::~MsgLog() 
{
  //destructor: clean up and reset instance pointer

  if (fRepetitions > 0) PrintRepetitions();

  for (int iType = kFatal; iType < kMaxType; iType++) {
    CloseFile(iType);
  }
  fflush(stderr);
  fflush(stdout);

  fgInstance = 0x0;

}

//_____________________________________________________________________________
MsgLog::MsgLog(const MsgLog& log) :
  fGlobalLogLevel(log.fGlobalLogLevel),
  fPrintRepetitions(log.fPrintRepetitions),
  fRepetitions(log.fRepetitions),
  fLastType(log.fLastType),
  fLastMessage(log.fLastMessage),
  fLastFunction(log.fLastFunction),
  fLastFile(log.fLastFile),
  fLastLine(log.fLastLine)
{

  //copy constructor
  std::cout << "Copy constructor not implemented" << std::endl;
  ::abort();

}

//_____________________________________________________________________________
MsgLog& MsgLog::operator = (const MsgLog& /* log */) 
{
  //assignment operator

  std::cout << "assignment operator not implemented" << std::endl;
  return *this;

}

//_____________________________________________________________________________
void MsgLog::ReadEnvSettings()
{
  //load settings from Configuration
  //Not yet implemented

  return;
}

//_____________________________________________________________________________
void MsgLog::EnableDebug(bool enabled)
{
  //enable or disable debug output
  fgDebugEnabled = enabled;
}

//_____________________________________________________________________________
void MsgLog::SetGlobalLogLevel(EType type) 
{
  // set the global debug level
  if (!fgInstance) new MsgLog;
  fgInstance->fGlobalLogLevel = type;
}

//_____________________________________________________________________________
int MsgLog::GetGlobalLogLevel()
{
  // get the global log level
  if (!fgInstance) new MsgLog;
  return fgInstance->fGlobalLogLevel;
}

std::string MsgLog::GetGlobalLogLevelString()
{
  // get the global log level
  if (!fgInstance) new MsgLog;
  switch(fgInstance->fGlobalLogLevel){
    case kFatal: return "kFatal";
    case kError: return "kError";
    case kWarning: return "kWarning";
    case kInfo: return "kInfo";
    case kDebug: return "kDebug";
    default: return "Unknown!";

  }
  return "Unknown!";
}


//_____________________________________________________________________________
void MsgLog::SetGlobalDebugLevel(int level)
{
  //set the global debug level
  if (!fgInstance) new MsgLog;
  if (level < -kDebugOffset) level = -kDebugOffset;
  fgInstance->fGlobalLogLevel = kDebugOffset + level;
}

//_____________________________________________________________________________
int MsgLog::GetGlobalDebugLevel()
{
  // get the global debug level
  if (!fgInstance) new MsgLog;
  return fgInstance->fGlobalLogLevel - kDebugOffset;
}

//_____________________________________________________________________________
void MsgLog::SetStandardOutput()
{
  // write all log messages to the standard output (stdout)

  if (!fgInstance) new MsgLog;
  for (int iType = kFatal; iType < kMaxType; iType++) {
    fgInstance->CloseFile(iType);
    fgInstance->fOutputTypes[iType] = 0;
  }
}

//_____________________________________________________________________________
void MsgLog::SetStandardOutput(EType type)
{
  // write log messages of the given type to the standard output
  // (stdout)
  
  if ((type < kFatal) || (type >= kMaxType)) return;
  if (!fgInstance) new MsgLog;
  fgInstance->CloseFile(type);
  fgInstance->fOutputTypes[type] = 0;
}

//_____________________________________________________________________________
void MsgLog::SetErrorOutput()
{
  // write all log messages to the error output (stderr)

  if (!fgInstance) new MsgLog;
  for (int iType = kFatal; iType < kMaxType; iType++) {
    fgInstance->CloseFile(iType);
    fgInstance->fOutputTypes[iType] = 1;
  }
}

//_____________________________________________________________________________
void MsgLog::SetErrorOutput(EType type)
{
  // write log messages of the given type to the error output (stderr)

  if ((type < kFatal) || (type >= kMaxType)) return;
  if (!fgInstance) new MsgLog;
  fgInstance->CloseFile(type);
  fgInstance->fOutputTypes[type] = 1;
}

//_____________________________________________________________________________
void MsgLog::SetFileOutput(const char* filename)
{
  // write all log messages to the given file
  
  if (!fgInstance) new MsgLog;
  for (int iType = kFatal; iType < kMaxType; iType++) {
    if ((fgInstance->fOutputTypes[iType] == 2) &&
        (fgInstance->fFileNames[iType].compare(filename) != 0)) {
      fgInstance->CloseFile(iType);
    }
    fgInstance->fOutputTypes[iType] = 2;
    fgInstance->fFileNames[iType] = filename;
    fgInstance->fOutputFiles[iType] = NULL;
    fgInstance->fOutputStreams[iType] = NULL;
  }
}

//_____________________________________________________________________________
void MsgLog::SetFileOutput(EType type, const char* filename)
{
  // write log messages of the given type to the given file

  if ((type < kFatal) || (type >= kMaxType)) return;
  if (!fgInstance) new MsgLog;
  if ((fgInstance->fOutputTypes[type] == 2) &&
      (fgInstance->fFileNames[type].compare(filename) != 0)) {
    fgInstance->CloseFile(type);
  }
  fgInstance->fOutputTypes[type] = 2;
  fgInstance->fFileNames[type] = filename;
  fgInstance->fOutputFiles[type] = NULL;
  fgInstance->fOutputStreams[type] = NULL;
}

//_____________________________________________________________________________
void MsgLog::CloseFile(int type)
{
  // close the file for the given type if needed

  if ((fOutputTypes[type] == 2) && fOutputFiles[type]) {
    bool closeFile = true;
    for (int iType = kFatal; iType < kMaxType; iType++) {
      if ((iType != type) && (fOutputFiles[iType] == fOutputFiles[type])) {
        closeFile = false;
      }
    }
    if (closeFile) {
      fclose(fOutputFiles[type]);
      std::ofstream* stream=reinterpret_cast<std::ofstream*>(fOutputStreams[type]);
      stream->close();
      delete fOutputStreams[type];
    }
  }
  fOutputFiles[type] = NULL;
  fOutputStreams[type] = NULL;
  fFileNames[type] = "";
  fOutputTypes[type] = 0;
}

//_____________________________________________________________________________
FILE* MsgLog::GetOutputStream(int type)
{
  // get the output stream for the given type of messages

  if (type > kDebug) type = kDebug;
  if (fOutputTypes[type] == 0) return stdout;
  else if (fOutputTypes[type] == 1) return stderr;
  else if (fOutputTypes[type] == 2) {
    if (!fOutputFiles[type]) {
      FILE* file = NULL;
      std::ostream* stream = NULL;
      if (!fFileNames[type].empty()) {
        for (int iType = kFatal; iType < kMaxType; iType++) {
          if ((iType != type) &&
              (fFileNames[iType].compare(fFileNames[type]) == 0) &&
              fOutputFiles[iType]) {
            file = fOutputFiles[iType];
            stream = fOutputStreams[iType];
            break;
          }
        }
        if (!file) {
          file = fopen(fFileNames[type].c_str(), "a");
          //stream = new std::ofstream(fFileNames[type].c_str(), std::ios_base::app);
          stream = new std::ofstream(fFileNames[type].c_str());
        }
      }
      fOutputFiles[type] = file;
      fOutputStreams[type] = stream;
      if (!file) CloseFile(type);
    }
    if (fOutputFiles[type]) return fOutputFiles[type];
  }

  return stdout;
}

//_____________________________________________________________________________
void MsgLog::Flush()
{
  // flush the output streams

  if (!fgInstance) new MsgLog;
  for (int iType = kFatal; iType < kMaxType; iType++) {
    if (fgInstance->fOutputFiles[iType]) {
      fflush(fgInstance->fOutputFiles[iType]);
      fgInstance->fOutputStreams[iType]->flush();
    }
  }
  fflush(stderr);
  fflush(stdout);
}

//_____________________________________________________________________________
void MsgLog::SetPrintType(short on)
{
  // switch on or off the printing of the message type for all message
  // types

  if (!fgInstance) new MsgLog;
  for (int iType = kFatal; iType < kMaxType; iType++) {
    fgInstance->fPrintType[iType] = on;
  }
}

//_____________________________________________________________________________
void MsgLog::SetPrintType(EType type, short on)
{
  // switch on or off the printing of the message type for the given
  // message type

  if ((type < kFatal) || (type >= kMaxType)) return;
  if (!fgInstance) new MsgLog;
  fgInstance->fPrintType[type] = on;
}

//_____________________________________________________________________________
void MsgLog::SetPrintLocation(bool on)
{
  // switch on or off the printing of the file name and line number
  // for all message types

  if (!fgInstance) new MsgLog;
  for (int iType = kFatal; iType < kMaxType; iType++) {
    fgInstance->fPrintLocation[iType] = on;
  }
}

//_____________________________________________________________________________
void MsgLog::SetPrintLocation(EType type, bool on)
{
  // switch on or off the printing of the file name and line number
  // for the given message type

  if ((type < kFatal) || (type >= kMaxType)) return;
  if (!fgInstance) new MsgLog;
  fgInstance->fPrintLocation[type] = on;
}

//_____________________________________________________________________________
void MsgLog::SetPrintPrefix(bool on)
{
  // switch on or off the printing of the file name and line number
  // for all message types

  if (!fgInstance) new MsgLog;
  for (int iType = kFatal; iType < kMaxType; iType++) {
    fgInstance->fPrintPrefix[iType] = on;
  }
}

//_____________________________________________________________________________
void MsgLog::SetPrintPrefix(EType type, bool on)
{
  // switch on or off the printing of the file name and line number
  // for the given message type

  if ((type < kFatal) || (type >= kMaxType)) return;
  if (!fgInstance) new MsgLog;
  fgInstance->fPrintPrefix[type] = on;
}
//_____________________________________________________________________________
void MsgLog::SetPrintRepetitions(bool on)
{
  // switch on or off the printing of the number of repetitions of a message
  // instead of repeating the same message

  if (!fgInstance) new MsgLog;
  if (!on && (fgInstance->fRepetitions > 0)) fgInstance->PrintRepetitions();
  fgInstance->fPrintRepetitions = on;
}

//_____________________________________________________________________________
const char* MsgLog::Test(const char* file, const char* function)
{

  const char* temp = function;
  std::string all(file);
  all += "-";
  all += temp;
  
  return all.c_str();

}

//_____________________________________________________________________________
void MsgLog::PrintMessage(int type, const std::string message,
                          const char* function, const char* file, int line)
{
  // print the given message

  // don't print the message if it is repeated
  if (fPrintRepetitions &&
      (fLastType == type) &&
      ((message != "") && (fLastMessage.compare(message) == 0)) &&
      ((function && (fLastFunction.compare(function) == 0)) ||
       (!function && fLastFunction.empty()))&&
      ((file && (fLastFile.compare(file) == 0)) ||
       (!file && fLastFile.empty())) &&
      (fLastLine == line)) {
    fRepetitions++;
    return;
  }

  // print number of repetitions
  if (fRepetitions > 0) PrintRepetitions();

  // remember this message
  fRepetitions = 0;
  fLastType = type;
  fLastMessage = message;
  fLastFunction = function;
  fLastFile = file;
  fLastLine = line;

  // print the message
  FILE* stream = GetOutputStream(type);
  static const char* typeNames[kMaxType] =
    {"Fatal", "Error", "Warning", "Info", "Debug"};

  std::string filename = get_filename(file);

  if (fPrintType[type]!=0) {
    if(fPrintType[type]==1)
      PrintString(type, stream, "%c-", typeNames[type][0]);
    else
      PrintString(type, stream, "%s-", typeNames[type]);
  }

  if(fPrintPrefix[type])   PrintString(type, stream, "%s-%s", filename.c_str(), function);
  if(fPrintLocation[type]) PrintString(type, stream, " (L%.0d)", line);
  if(message != "")        PrintString(type, stream, ": %s", message.c_str());  
  PrintString(type, stream, "\n");
  if (fCallBacks[type]) (*(fCallBacks[type]))((EType)type, NULL);
}

//_____________________________________________________________________________
void MsgLog::PrintRepetitions()
{
  // print number of repetitions

  PrintString(fLastType, GetOutputStream(fLastType), " <message repeated %d time%s>\n",
	      fRepetitions, (fRepetitions > 1) ? "s" : "");
  if (fCallBacks[fLastType]) (*(fCallBacks[fLastType]))((EType)fLastType, NULL);
}

//_____________________________________________________________________________
void MsgLog::Message(int level, const std::string message,
		    const char* function, const char* file, int line)
{
  // print a log message

  if (!fgInstance) new MsgLog;

  // get the message type
  int type = level;
  if (type >= kMaxType) type = kMaxType - 1;

  // print the message if the debug level allows
  if (level <= fgInstance->GetGlobalLogLevel()) {
    fgInstance->PrintMessage(type, message, function, file, line);
  }

  // abort in case of a fatal message
  if (type == kFatal) {
    delete fgInstance;
    ::abort();
  }
}

//_____________________________________________________________________________
//void MsgLog::IsDebugEnabled()
//{
// return fgDebugEnabled;
//}

//_____________________________________________________________________________
void MsgLog::Debug(int level, const std::string message,
		      const char* function, const char* file, int line)
{
  // print a debug message

  if (level == 0) level = 1;
  level += kDebugOffset;
  Message(level, message, function, file, line);
}


//_____________________________________________________________________________
int MsgLog::RedirectStdoutTo(EType type, int level, const char* function,
                               const char* file, int line, bool print)
{
  // redirect the standard output to the stream of the given type

  if (!fgInstance) new MsgLog;
  return fgInstance->RedirectTo(stdout, type, level, function, file, line, print);
}

//_____________________________________________________________________________
int MsgLog::RedirectStderrTo(EType type, int level, const char* function,
                               const char* file, int line, bool print)
{
  // redirect the standard error output to the stream of the given
  // type

  if (!fgInstance) new MsgLog;
  return fgInstance->RedirectTo(stderr, type, level, function, file, line, print);

}

//_____________________________________________________________________________
int MsgLog::RedirectTo(FILE* stream, EType type, int level, 
			  const char* function, const char* file, int line, 
			  bool print)
{
  // redirect the standard (error) output stream to the stream of the
  // given type

  // get the original file descriptor to be able to restore it later
  int original = dup(fileno(stream));
  fflush(stream);

  // flush the stream of the selected type
  FILE* newStream = GetOutputStream(type);
  fflush(newStream);

  // redirect stream
  if ((type == kDebug) && (level > 0)) level--;
  if (type + level > GetGlobalLogLevel()) { // /dev/null
    freopen("/dev/null", "a", stream);
  } else if (fOutputTypes[type] == 0) {         // stdout
    if (stream != stdout) dup2(fileno(stdout), fileno(stream));
  } else if (fOutputTypes[type] == 1) {         // stderr
    if (stream != stderr) dup2(fileno(stderr), fileno(stream));
  } else if (fOutputTypes[type] == 2) {         // file
    freopen(fFileNames[type].c_str(), "a", stream);
  } else if (fOutputTypes[type] == 3) {         // external C++ stream
    // redirection is not possible for external C++ streams
  }

  // print information
  if (print) {
    PrintMessage(type, NULL, function, file, line);
    fflush(newStream);
  }

  return original;
}

//_____________________________________________________________________________
void MsgLog::RestoreStdout(int original)
{
  // restore the standard output
  fflush(stdout);
  dup2(original, fileno(stdout));
  close(original);
}

//_____________________________________________________________________________
void MsgLog::RestoreStderr(int original)
{
  // restore the standard error output

  fflush(stderr);
  dup2(original, fileno(stderr));
  close(original);
}

//_____________________________________________________________________________
std::ostream& MsgLog::Stream(EType type, int level, const char* function, 
			   const char* file, int line)
{
  // get the stream object for the given output type

  if (!fgInstance) new MsgLog;
  return fgInstance->GetStream(type, level, function, file, line);
}

//_____________________________________________________________________________
std::ostream& MsgLog::GetStream(EType type, int level, const char* function, 
			      const char* file, int line)
{
  // get the stream object for the given output type

  if ((type == kDebug) && (level > 0)) level--;
  bool noOutput = (type + level > GetGlobalLogLevel());

  if (!noOutput) {
    PrintMessage(type, NULL, function, file, line);
  }
  fflush(GetOutputStream(type));

  static std::ofstream nullStream("/dev/null");
  if (noOutput) {
    return nullStream;
  } else if (fOutputTypes[type] == 0) {
    return std::cout;
  } else if (fOutputTypes[type] == 1) {
    return std::cerr;
  } else if (fOutputTypes[type] == 2) {
    return *fOutputStreams[type];
  } else if (fOutputTypes[type] == 3) {
    return *fOutputStreams[type];
  }

  return nullStream;
}

//_____________________________________________________________________________
void  MsgLog::SetStreamOutput(std::ostream* stream)
{
  // set an external stream as target for log messages of all types
  // the external stream is completely handled by the caller, the 
  // MsgLog class just writes to it

  for (int iType = kFatal; iType < kMaxType; iType++) {
    SetStreamOutput((MsgLog::EType)iType, stream);
  }
}

//_____________________________________________________________________________
void  MsgLog::SetStreamOutput(EType type, std::ostream* stream)
{
  // set an external stream as target for log messages of the given type
  // the external stream is completely handled by the caller, the
  // MsgLog class just writes to it

  if ((type < kFatal) || (type >= kMaxType)) return;
  if (!fgInstance) new MsgLog;
  if (fgInstance->fOutputTypes[type] == 2) {
    fgInstance->CloseFile(type);
  }
  fgInstance->fOutputTypes[type] = 3;
  fgInstance->fFileNames[type] = "";
  fgInstance->fOutputFiles[type] = NULL;
  fgInstance->fOutputStreams[type] = stream;
}

//_____________________________________________________________________________
void  MsgLog::SetLogNotification(MsgLogNotification pCallBack)
{
  // set a notification callback function for log messages of all
  // types

  for (int iType = kFatal; iType < kMaxType; iType++) {
    SetLogNotification((MsgLog::EType)iType, pCallBack);
  }
}

//_____________________________________________________________________________
void  MsgLog::SetLogNotification(EType type, MsgLogNotification pCallBack)
{
  // set a notifications call back function for log messages of all types
  // the callback fuction is invoked whenever an output was written
  // Note: does not work for c++ streamer classes, the external stream
  // has to handle this directly (e.g. custom implementation of endl)  

  if ((type < kFatal) || (type >= kMaxType)) return;
  if (!fgInstance) new MsgLog;
  fgInstance->fCallBacks[type]=pCallBack;
}

//_____________________________________________________________________________
void  MsgLog::PrintString(int type, FILE* stream, const char* format, ...)
{
  // this is the general method to print a log message using variadac args
  // to the FILE* like (C - like) streams, e.g. stdout, stderr, or files
  // opened by fopen.
  // Only in case of an external c++ ostream type output, the message is
  // written to that stream and the notification callback is called.
  // The message is printed by a normal vfprintf function otherwise

  if (format==NULL) return;

  va_list ap;
  va_start(ap, format);
  if (fOutputTypes[type] != 3) {
    if (stream!=NULL) {
      vfprintf(stream, format, ap);
    }
  } else {
    // build the string and write everthing to the corresponding
    // ostream
    char* tgt = new char[sizeof(format)*10];
    va_list bap;
    va_copy(bap, ap);

    int iResult=0;
    while (1) {
       iResult=vsnprintf(tgt, strlen(tgt), format, ap);
      if (iResult==-1) {
        iResult=strlen(tgt)*2;
      } else if ((unsigned int)iResult<strlen(tgt)) {
        break;
      }
      if (iResult<10000) {
	tgt = (char*)realloc(tgt,iResult+1);
        va_end(ap);
        va_copy(ap, bap);
      } else
	{
	  tgt[strlen(tgt)-1]=0;
	  break;
	}
    }
    va_end(bap);

    if (fOutputStreams[type]) {
      *(fOutputStreams[type]) << tgt;
    }
  delete [] tgt;
  }
  va_end(ap);
}

//_____________________________________________________________________________
const std::string  MsgLog::Form(const char *fmt, ...)
{
  // Formats a string using a printf style format descriptor.
  // Existing string contents will be overwritten.

  std::string mystr;
  char* tmp = NULL;
  size_t buflen = 20 * strlen(fmt);  //hopefully big enough
  if((tmp = (char*)malloc(buflen)) == NULL)
    return mystr; //return 0;

  va_list ap;
  int iResult = 0;

  while (1) {
    /* Try to print in the allocated space */
    va_start(ap,fmt);
    iResult=vsnprintf(tmp, buflen, fmt, ap);
    va_end(ap);
    /* if that worked, return the string */
    if (iResult > -1 && (unsigned int)iResult < buflen) {
      mystr = tmp;
      free(tmp);
      return mystr;
      //return mystr.c_str();
    } else {
      if( iResult > -1)
	buflen = iResult + 1;
      else
	buflen *= 2;
      if ((tmp = (char*)realloc(tmp,buflen)) == NULL)
        return mystr; //return 0;
    }

  }

}

/*/_______________________________________________________________________________
template<typename ... Args>
const std::string MsgLog::FormStr(const std::string& format, Args ... args )
  {
    size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    std::unique_ptr<char[]> buf( new char[ size ] ); 
    snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
  }
*/