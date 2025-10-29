"""
Retry Handler
Implements retry logic with exponential backoff

Handles transient failures in:
- API calls
- LLM requests
- File operations
"""

import asyncio
import time
from typing import Any, Callable, Optional
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class RetryHandler:
    """
    Handles retry logic with exponential backoff
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        """
        Initialize retry handler
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            exponential_base: Base for exponential backoff
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        
        logger.info(f"RetryHandler initialized with max_retries={max_retries}")
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic
        
        Args:
            func: Function to execute (can be sync or async)
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            Exception if all retries exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Check if function is async
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"Retry succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_retries + 1} attempts failed. "
                        f"Last error: {str(e)}"
                    )
        
        # All retries exhausted
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for exponential backoff
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)
    
    def retry_decorator(
        self,
        max_retries: Optional[int] = None,
        exceptions: tuple = (Exception,)
    ):
        """
        Decorator for adding retry logic to functions
        
        Args:
            max_retries: Override max retries for this function
            exceptions: Tuple of exceptions to catch and retry
            
        Example:
            @retry_handler.retry_decorator(max_retries=5)
            async def my_function():
                ...
        """
        retries = max_retries if max_retries is not None else self.max_retries
        
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(retries + 1):
                    try:
                        if asyncio.iscoroutinefunction(func):
                            return await func(*args, **kwargs)
                        else:
                            return func(*args, **kwargs)
                    
                    except exceptions as e:
                        last_exception = e
                        
                        if attempt < retries:
                            delay = self._calculate_delay(attempt)
                            logger.warning(
                                f"{func.__name__} attempt {attempt + 1} failed: {e}. "
                                f"Retrying in {delay:.1f}s..."
                            )
                            await asyncio.sleep(delay)
                
                raise last_exception
            
            return wrapper
        return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern for failing fast when service is down
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to track for failures
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        
        logger.info(f"CircuitBreaker initialized with threshold={failure_threshold}")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker
        
        States:
        - closed: Normal operation, calls pass through
        - open: Circuit broken, calls fail immediately
        - half_open: Testing if service recovered
        """
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                logger.info("Circuit breaker entering half-open state")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == "half_open":
                self._reset()
                logger.info("Circuit breaker reset to CLOSED state")
            
            self.failure_count = 0
            return result
            
        except self.expected_exception as e:
            self._record_failure()
            raise e
    
    def _record_failure(self) -> None:
        """Record a failure and potentially open circuit"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.error(
                f"Circuit breaker OPENED after {self.failure_count} failures"
            )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.recovery_timeout
    
    def _reset(self) -> None:
        """Reset circuit breaker to closed state"""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None